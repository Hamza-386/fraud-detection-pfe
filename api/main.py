"""
FastAPI - Service de Détection de Fraude en Temps Réel (sécurisé)
Projet PFE - Hamza Chaabene - TEK-UP / Capgemini

Charge le modèle de production produit par la CELL 18 du notebook
et expose des endpoints REST protégés pour la prédiction.

Modèle de production (clé "production_model" de thresholds.json,
modifiable via la variable d'environnement FRAUD_MODEL) :
- "stacking" : LightGBM + XGBoost + CatBoost (modèles complets de la
  CELL 13B) combinés par une régression logistique
  (stack_*_model.pkl + catboost_model.cbm, voir stacking_config.json)
- "catboost" : CatBoost seul (catboost_model.cbm)

Les features absentes de la requête sont complétées avec les valeurs
d'entraînement de preprocessing.pkl (médiane d'entraînement, ou taux
de fraude global pour les colonnes target-encodées).

Sécurité:
- Authentification par clé API (header X-API-Key), lue depuis un
  fichier .env (voir .env.example) via python-dotenv — jamais
  codée en dur dans le code source.
- Limitation du débit (rate limiting) par IP
- Validation stricte des entrées (valeurs finies, dict non vide)

Décision à trois niveaux (au lieu d'un seuil binaire) :
- proba <  seuil_review (profit-optimisé)      -> auto_approve
- seuil_review <= proba < seuil_block (F1-opt.) -> manual_review
- proba >= seuil_block                          -> auto_block

Lancer avec:
    uvicorn api.main:app --reload --port 8000

Configuration de la clé API (recommandé) :
    1. Copier .env.example vers .env : cp .env.example .env
    2. Éditer .env et remplacer la valeur de FRAUD_API_KEY
    3. Le fichier .env n'est jamais versionné (voir .gitignore)
"""
import json
import logging
import math
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ============================================
# Journalisation (traçabilité des prédictions)
# ============================================
# Exigence EF-09 / section 5.2 du Cahier des Charges : chaque
# prédiction doit être journalisée pour audit. Un fichier de logs
# rotatif simple suffit pour ce prototype ; en production, ceci
# serait typiquement redirigé vers un système centralisé (ELK,
# CloudWatch, etc.).
LOG_PATH = os.getenv("FRAUD_LOG_PATH", "predictions.log")

logger = logging.getLogger("fraud_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    logger.addHandler(_file_handler)


def _log_prediction(endpoint: str, proba: float, decision: str) -> None:
    """Journalise une prédiction individuelle pour audit ultérieur."""
    logger.info(
        json.dumps({
            "endpoint": endpoint,
            "fraud_probability": round(proba, 4),
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    )

# Charge les variables du fichier .env (s'il existe) dans
# les variables d'environnement du processus, AVANT de les lire
# ci-dessous avec os.getenv(). N'écrase pas une variable
# d'environnement déjà définie par le système.
load_dotenv()

# ============================================
# Configuration
# ============================================
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(
    os.path.dirname(__file__), "..", "models"
))

DEFAULT_API_KEY = "dev-only-change-me"
API_KEY = os.getenv("FRAUD_API_KEY", DEFAULT_API_KEY)
MAX_BATCH_SIZE = 500
# Temps réel : état en ligne (historique par carte / client) construit
# par scripts/build_online_state.py. Absent -> /predict/raw indisponible.
ONLINE_STATE_PATH = os.getenv(
    "FRAUD_ONLINE_STATE", os.path.join(MODELS_DIR, "online_state.pkl")
)
# /predict/raw est appelé pour CHAQUE paiement par le serveur de
# paiement (une seule IP) : sa limite est celle d'un flux, pas d'un humain.
STREAM_RATE_LIMIT = os.getenv("FRAUD_STREAM_RATE_LIMIT", "6000/minute")

limiter = Limiter(key_func=get_remote_address)

# État global chargé au démarrage
# `model` = CatBoost : modèle de production en mode "catboost",
# modèle de base (et modèle expliqué par SHAP) en mode "stacking".
model: Optional[CatBoostClassifier] = None
feature_names: List[str] = []
thresholds: Dict[str, float] = {}
explainer: Optional[shap.TreeExplainer] = None
MAX_EXPLAIN_TOP_N = 20

SUPPORTED_MODELS = ("stacking", "catboost")
production_model: str = "catboost"
stack_base_models: Dict[str, object] = {}   # "lgb" / "xgb" (CatBoost = model)
stack_meta_model = None
stack_input_order: List[str] = []
default_values: Dict[str, float] = {}       # valeurs de complétion des features
# Moteur de features en ligne (realtime.engine.OnlineFeatureEngine) ; son
# état est partagé entre requêtes, d'où le verrou.
online_engine = None
engine_lock = threading.Lock()

# Bornes du système de décision à 3 niveaux (calculées au démarrage
# à partir de thresholds.json, avec repli si les clés sont absentes)
REVIEW_LOWER: float = 0.3   # sous ce seuil : auto_approve
BLOCK_UPPER: float = 0.7    # au-dessus : auto_block


def _load_default_values(path: str) -> Dict[str, float]:
    """
    Valeurs utilisées pour compléter les features absentes d'une
    requête, identiques à celles du notebook :
    - médiane d'entraînement (CELL 10) pour les colonnes imputées ;
    - taux de fraude global des encodeurs (CELL 13B) pour les colonnes
      target-encodées et uid_target_enc — c'est exactement la valeur
      qu'obtient une clé inconnue.
    Les autres colonnes (jamais manquantes à l'entraînement) -> 0.0.
    """
    if not os.path.exists(path):
        print(
            f"⚠️  {path} introuvable — les features absentes seront "
            f"complétées avec 0.0."
        )
        return {}
    preprocessing = joblib.load(path)
    defaults = {
        col: float(val)
        for col, val in preprocessing["train_medians"].items()
    }
    for col, encoder in preprocessing["te_encoders"].items():
        target_col = "uid_target_enc" if col == "uid" else col
        defaults[target_col] = float(encoder[2])   # global_mean
    print(f"✅ Valeurs de complétion chargées ({len(defaults)} features)")
    return defaults


def _load_stacking() -> None:
    """Charge les modèles de base LightGBM / XGBoost et le méta-modèle."""
    global stack_base_models, stack_meta_model, stack_input_order

    config_path = os.path.join(MODELS_DIR, "stacking_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"❌ {config_path} introuvable: le mode 'stacking' exige les "
            f"fichiers produits par la CELL 18 du notebook."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # CatBoost (déjà chargé dans `model`) est le 3e modèle de base
    if config["base_models"].get("catboost") != "catboost_model.cbm":
        raise RuntimeError("❌ stacking_config.json: modèle CatBoost inattendu.")

    stack_base_models = {
        name: joblib.load(os.path.join(MODELS_DIR, filename))
        for name, filename in config["base_models"].items()
        if name != "catboost"
    }
    stack_meta_model = joblib.load(
        os.path.join(MODELS_DIR, config["meta_model"])
    )
    stack_input_order = list(config["meta_input_order"])

    expected = set(stack_base_models) | {"catboost"}
    if set(stack_input_order) != expected:
        raise RuntimeError(
            f"❌ stacking_config.json: meta_input_order "
            f"{stack_input_order} ne correspond pas aux modèles {expected}."
        )
    print(
        f"✅ Stacking chargé : {' + '.join(stack_input_order)} "
        f"-> {type(stack_meta_model).__name__}"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Gestionnaire de cycle de vie moderne de FastAPI (remplace
    @app.on_event("startup"), déprécié). Le code avant `yield`
    s'exécute au démarrage ; après `yield`, à l'arrêt (rien à faire
    ici, mais le point d'extension existe si besoin plus tard, ex.
    fermeture propre d'une connexion base de données).
    """
    global model, feature_names, thresholds, explainer
    global REVIEW_LOWER, BLOCK_UPPER
    global production_model, stack_base_models, stack_meta_model
    global stack_input_order, default_values

    if API_KEY == DEFAULT_API_KEY:
        print(
            "⚠️  ATTENTION: FRAUD_API_KEY non définie — utilisation "
            "d'une clé par défaut non sécurisée. Définissez la "
            "variable d'environnement FRAUD_API_KEY avant tout usage "
            "hors développement local."
        )

    model_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    features_path = os.path.join(MODELS_DIR, "feature_names.json")
    thresholds_path = os.path.join(MODELS_DIR, "thresholds.json")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"❌ Modèle introuvable: {model_path}\n"
            f"Vérifiez que le dossier 'models/' (téléchargé depuis "
            f"Kaggle) est bien placé à la racine du projet."
        )

    model = CatBoostClassifier()
    model.load_model(model_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    print(f"✅ Modèle CatBoost chargé ({len(feature_names)} features)")

    default_values = _load_default_values(
        os.path.join(MODELS_DIR, "preprocessing.pkl")
    )

    production_model = os.getenv(
        "FRAUD_MODEL", thresholds.get("production_model", "catboost")
    ).strip().lower()
    if production_model not in SUPPORTED_MODELS:
        raise RuntimeError(
            f"❌ Modèle de production inconnu: '{production_model}' "
            f"(valeurs acceptées: {', '.join(SUPPORTED_MODELS)})"
        )
    if production_model == "stacking":
        _load_stacking()

    print(f"✅ Modèle de production : {production_model}")
    print(
        "✅ Threshold de production (profit-optimisé): "
        f"{thresholds.get(f'{production_model}_profit_optimized', 0.5)}"
    )

    # REVIEW_LOWER = seuil profit-optimisé (début de la zone de doute)
    # BLOCK_UPPER  = seuil F1-optimisé, historiquement plus élevé
    #                (haute confiance -> blocage automatique)
    # Les deux sont choisis sur la validation par le notebook, pour
    # le modèle de production lui-même ("stacking" ou "catboost").
    REVIEW_LOWER = thresholds.get(f"{production_model}_profit_optimized", 0.3)
    BLOCK_UPPER = thresholds.get(production_model, 0.7)
    if BLOCK_UPPER <= REVIEW_LOWER:
        # Garde-fou : si les seuils sauvegardés sont incohérents,
        # on garantit un écart minimal pour que la zone de revue
        # manuelle existe réellement.
        BLOCK_UPPER = min(REVIEW_LOWER + 0.3, 0.95)
    print(
        f"✅ Décision 3 niveaux : auto_approve < {REVIEW_LOWER:.2f} "
        f"<= manual_review < {BLOCK_UPPER:.2f} <= auto_block"
    )

    explainer = shap.TreeExplainer(model)
    print("✅ Explainer SHAP initialisé (TreeExplainer sur CatBoost)")

    global online_engine
    if os.path.exists(ONLINE_STATE_PATH):
        with open(ONLINE_STATE_PATH, "rb") as f:
            online_engine = pickle.load(f)
        if list(online_engine.feature_names) != list(feature_names):
            raise RuntimeError(
                "❌ online_state.pkl ne correspond pas à feature_names.json : "
                "relancez scripts/build_online_state.py."
            )
        print(f"✅ Moteur temps réel chargé : {online_engine.state_summary()}")
    else:
        online_engine = None
        print(
            f"ℹ️  {ONLINE_STATE_PATH} absent : /predict/raw désactivé "
            f"(lancez scripts/build_online_state.py)."
        )

    yield  # L'API est prête et sert les requêtes ici

    # --- Code d'arrêt (shutdown) ---
    print("👋 Arrêt de l'API Fraud Detection")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "API de détection de fraude en temps réel (stacking LightGBM + "
        "XGBoost + CatBoost, ou CatBoost seul) "
        "- Projet PFE (endpoints protégés par clé API)"
    ),
    version="1.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
def root():
    """Redirige les visiteurs humains vers la documentation Swagger."""
    return {"message": "Fraud Detection API", "docs": "/docs"}


@app.exception_handler(Exception)
async def _hide_internal_errors(request: Request, exc: Exception):
    """
    Évite de renvoyer une trace de pile Python brute (qui pourrait
    exposer des détails internes) pour toute erreur inattendue non
    déjà gérée par une HTTPException explicite.
    """
    from fastapi.responses import JSONResponse
    logger.error(f"Erreur non gérée sur {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur."},
    )


# ============================================
# Sécurité : authentification par clé API
# ============================================
def verify_api_key(x_api_key: str = Header(default=None)):
    """Dépendance FastAPI vérifiant l'en-tête X-API-Key."""
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="En-tête 'X-API-Key' manquant.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide.")
    return True


# ============================================
# Schémas Pydantic (avec validation stricte)
# ============================================
class Transaction(BaseModel):
    """
    Une transaction déjà prétraitée : dictionnaire {nom_feature: valeur}.
    Les features manquantes par rapport à feature_names.json sont
    automatiquement complétées avec les valeurs d'entraînement
    (preprocessing.pkl : médiane, ou taux de fraude global pour les
    colonnes target-encodées).

    Important : card1, card2, addr1, P_emaildomain et uid_target_enc
    sont des valeurs TARGET-ENCODÉES (taux de fraude lissé, entre 0
    et 1), pas les identifiants bruts.

    Note (limitation assumée du projet): le pipeline complet de
    feature engineering (UID, target encoding, agrégations, lag/rolling)
    nécessite l'historique complet des transactions et n'est pas
    reproduit ici en temps réel. L'API attend donc un vecteur de
    features déjà calculé (ex: par un job de feature engineering
    en amont), conformément au périmètre du stage PFE.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire des features prétraitées",
        examples=[{"TransactionAmt": 125.50, "card1_freq": 340}],
    )

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v):
        if not v:
            raise ValueError("Le dictionnaire de features est vide.")
        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(
                    f"Valeur non finie (NaN/inf) pour la feature "
                    f"'{key}'."
                )
        return v


class BatchTransactions(BaseModel):
    transactions: List[Dict[str, float]]

    @field_validator("transactions")
    @classmethod
    def _validate_batch(cls, v):
        if not v:
            raise ValueError("La liste de transactions est vide.")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Lot trop volumineux ({len(v)} transactions) — "
                f"limite: {MAX_BATCH_SIZE}."
            )
        for i, feats in enumerate(v):
            for key, val in feats.items():
                if not math.isfinite(val):
                    raise ValueError(
                        f"Transaction {i}: valeur non finie pour "
                        f"'{key}'."
                    )
        return v


RAW_REQUIRED = ("TransactionID", "TransactionDT", "TransactionAmt", "card1")


class RawTransaction(BaseModel):
    """
    Une transaction BRUTE, telle qu'elle arrive du système de paiement :
    colonnes de train_transaction.csv + train_identity.csv (sans isFraud).
    Les colonnes absentes ou nulles sont traitées comme manquantes, comme
    dans le notebook. Toutes les features (UID, historique de la carte et
    du client, target encoding, ...) sont calculées en ligne par l'API.
    """
    transaction: Dict[str, Union[int, float, str, None]] = Field(
        ...,
        examples=[{"TransactionID": 3577540, "TransactionDT": 15811131,
                   "TransactionAmt": 59.0, "ProductCD": "W", "card1": 9500,
                   "card2": 321.0, "card4": "visa", "card6": "debit",
                   "addr1": 204.0, "P_emaildomain": "gmail.com"}],
    )

    @field_validator("transaction")
    @classmethod
    def _validate_raw(cls, v):
        for key in RAW_REQUIRED:
            if v.get(key) is None:
                raise ValueError(f"Champ obligatoire manquant : '{key}'.")
        for key in ("TransactionDT", "TransactionAmt", "card1"):
            val = v[key]
            if isinstance(val, str) or not math.isfinite(val):
                raise ValueError(f"'{key}' doit être un nombre fini.")
        if v["TransactionAmt"] < 0 or v["TransactionDT"] < 0:
            raise ValueError("TransactionAmt et TransactionDT doivent être >= 0.")
        for key, val in v.items():
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError(f"Valeur non finie pour '{key}'.")
        return v


class LabelFeedback(BaseModel):
    """Label confirmé a posteriori (ex : rétrofacturation), qui met à
    jour l'historique de fraude du client, de la carte, ..."""
    transaction_id: Union[int, str]
    is_fraud: int = Field(..., ge=0, le=1)


# ============================================
# Fonctions utilitaires
# ============================================
def _complete_row(feats: Dict[str, float]) -> Dict[str, float]:
    """Vecteur complet : valeur fournie, sinon valeur d'entraînement."""
    return {
        col: feats.get(col, default_values.get(col, 0.0))
        for col in feature_names
    }


def _build_feature_vector(feats: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([_complete_row(feats)], columns=feature_names)


def _build_batch_matrix(list_feats: List[Dict[str, float]]) -> pd.DataFrame:
    rows = [_complete_row(feats) for feats in list_feats]
    return pd.DataFrame(rows, columns=feature_names)


def _base_predict(base_model, A: np.ndarray) -> np.ndarray:
    """Probabilité de fraude d'un modèle de base via son booster natif
    (LightGBM : booster_.predict ; XGBoost : inplace_predict), sans la
    surcouche scikit-learn. Résultat identique à predict_proba[:, 1]."""
    if hasattr(base_model, "booster_"):          # LGBMClassifier
        return base_model.booster_.predict(A)
    if hasattr(base_model, "get_booster"):       # XGBClassifier
        return base_model.get_booster().inplace_predict(A)
    return base_model.predict_proba(A)[:, 1]


def _predict_proba(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Probabilité de fraude du modèle de production.
    - "catboost" : sortie de CatBoost.
    - "stacking" : probabilités des 3 modèles de base, combinées par
      le méta-modèle dans l'ordre de stacking_config.json.
    """
    # Tableau numpy plutôt que DataFrame : mêmes probabilités, mais la
    # validation pandas de LightGBM / XGBoost coûtait ~30 ms par appel.
    A = X if isinstance(X, np.ndarray) else X.to_numpy(dtype=np.float64)
    proba_cb = model.predict_proba(A)[:, 1]
    if production_model != "stacking":
        return proba_cb

    base_probas = {"catboost": proba_cb}
    for name, base_model in stack_base_models.items():
        base_probas[name] = _base_predict(base_model, A)
    meta_input = np.column_stack(
        [base_probas[name] for name in stack_input_order]
    ).astype(np.float64)
    return stack_meta_model.predict_proba(meta_input)[:, 1]


def _decision(proba: float) -> Dict[str, str]:
    """
    Détermine l'action opérationnelle à trois niveaux, au lieu d'un
    simple seuil binaire is_fraud true/false :
    - auto_approve  : risque faible, transaction traitée normalement
    - manual_review : zone d'incertitude, à faire vérifier par un
                       analyste avant décision finale
    - auto_block     : confiance élevée de fraude, blocage immédiat
    """
    if proba >= BLOCK_UPPER:
        return {"decision": "auto_block", "risk_level": "high"}
    if proba >= REVIEW_LOWER:
        return {"decision": "manual_review", "risk_level": "medium"}
    return {"decision": "auto_approve", "risk_level": "low"}


# ============================================
# Endpoints
# ============================================
@app.get("/health")
def health():
    """
    Vérifie que le service et le modèle sont opérationnels.
    Volontairement PUBLIC (pas de clé requise) pour permettre un
    monitoring externe simple (ex: load balancer, uptime checker).
    """
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "production_model": production_model,
        "n_features_expected": len(feature_names),
        "realtime_enabled": online_engine is not None,
    }


@app.get("/model/info", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def model_info(request: Request):
    """Retourne les métadonnées du modèle chargé. (protégé)"""
    if model is None:
        raise HTTPException(503, "Modèle non chargé")
    info = {
        "production_model": production_model,
        "model_type": (
            "Stacking (LightGBM + XGBoost + CatBoost -> "
            f"{type(stack_meta_model).__name__})"
            if production_model == "stacking" else "CatBoost"
        ),
        "n_features": len(feature_names),
        "production_threshold": REVIEW_LOWER,
        "f1_optimized_threshold": thresholds.get(production_model),
        "decision_thresholds": {
            "auto_approve_below": REVIEW_LOWER,
            "auto_block_at_or_above": BLOCK_UPPER,
        },
        "explained_model": "catboost",
        "all_thresholds": thresholds,
    }
    if production_model == "stacking":
        info["stacking"] = {
            "meta_input_order": stack_input_order,
            "meta_coefficients": [
                round(float(c), 6) for c in stack_meta_model.coef_[0]
            ],
            "meta_intercept": round(float(stack_meta_model.intercept_[0]), 6),
        }
    return info


@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def predict(request: Request, tx: Transaction):
    """Prédit la probabilité de fraude pour UNE transaction. (protégé)"""
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    X = _build_feature_vector(tx.features)
    proba = float(_predict_proba(X)[0])
    decision_info = _decision(proba)
    _log_prediction("/predict", proba, decision_info["decision"])

    return {
        "fraud_probability": round(proba, 4),
        **decision_info,
        "thresholds": {
            "auto_approve_below": REVIEW_LOWER,
            "auto_block_at_or_above": BLOCK_UPPER,
        },
    }


@app.post("/predict/explain", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
def predict_explain(
    request: Request,
    tx: Transaction,
    top_n: int = Query(
        default=5, ge=1, le=MAX_EXPLAIN_TOP_N,
        description="Nombre de features les plus contributives à "
                     "retourner",
    ),
):
    """
    Prédit la probabilité de fraude ET explique la décision via les
    valeurs SHAP (contribution de chaque feature à cette transaction
    précise). Plus coûteux que /predict — usage recommandé pour la
    revue manuelle des alertes, pas pour le scoring en masse.

    En mode "stacking", la probabilité et la décision sont celles du
    stacking, mais les valeurs SHAP expliquent le modèle de base
    CatBoost (le méta-modèle n'a pas d'explication SHAP exacte).
    """
    if model is None or explainer is None:
        raise HTTPException(503, "Modèle ou explainer non chargé")

    X = _build_feature_vector(tx.features)
    proba = float(_predict_proba(X)[0])
    decision_info = _decision(proba)
    _log_prediction("/predict/explain", proba, decision_info["decision"])

    shap_values = explainer.shap_values(X)
    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    top_features = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "effect": "augmente_le_risque" if val > 0
                      else "diminue_le_risque",
        }
        for feat, val in contributions[:top_n]
    ]

    return {
        "fraud_probability": round(proba, 4),
        **decision_info,
        "thresholds": {
            "auto_approve_below": REVIEW_LOWER,
            "auto_block_at_or_above": BLOCK_UPPER,
        },
        "explained_model": "catboost",
        "catboost_probability": round(float(model.predict_proba(X)[0, 1]), 4),
        "base_value": round(float(explainer.expected_value), 4),
        "top_contributing_features": top_features,
    }


@app.post("/predict/batch", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def predict_batch(request: Request, batch: BatchTransactions):
    """Prédit la probabilité de fraude pour une LISTE de transactions.
    (protégé, limité à MAX_BATCH_SIZE transactions par appel)"""
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    X = _build_batch_matrix(batch.transactions)
    probas = _predict_proba(X)

    results = [
        {
            "index": i,
            "fraud_probability": round(float(p), 4),
            **_decision(float(p)),
        }
        for i, p in enumerate(probas)
    ]
    n_total = len(results)
    n_auto_block = sum(r["decision"] == "auto_block" for r in results)
    n_manual_review = sum(
        r["decision"] == "manual_review" for r in results
    )
    n_auto_approve = sum(
        r["decision"] == "auto_approve" for r in results
    )
    logger.info(json.dumps({
        "endpoint": "/predict/batch",
        "n_transactions": n_total,
        "n_auto_approve": n_auto_approve,
        "n_manual_review": n_manual_review,
        "n_auto_block": n_auto_block,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))
    return {
        "n_transactions": n_total,
        "n_auto_approve": n_auto_approve,
        "n_manual_review": n_manual_review,
        "n_auto_block": n_auto_block,
        "manual_review_rate": round(n_manual_review / n_total, 4),
        "thresholds": {
            "auto_approve_below": REVIEW_LOWER,
            "auto_block_at_or_above": BLOCK_UPPER,
        },
        "results": results,
    }


# ============================================
# Temps réel : transactions brutes
# ============================================
def _require_engine():
    if model is None:
        raise HTTPException(503, "Modèle non chargé")
    if online_engine is None:
        raise HTTPException(
            503,
            "Moteur temps réel non chargé (models/online_state.pkl absent : "
            "lancez scripts/build_online_state.py).",
        )


@app.post("/predict/raw", dependencies=[Depends(verify_api_key)])
@limiter.limit(STREAM_RATE_LIMIT)
def predict_raw(request: Request, tx: RawTransaction):
    """
    Score une transaction BRUTE au moment où elle arrive. (protégé)

    L'API calcule elle-même les 512 features à partir de son état en
    ligne (historique de la carte, du client reconstitué, du domaine
    e-mail, ...), exactement comme le notebook, puis met cet état à jour
    avec la transaction. Aucun label n'est nécessaire à ce moment : les
    labels arrivent plus tard via /feedback.
    """
    _require_engine()
    t0 = time.perf_counter()
    raw = tx.transaction
    with engine_lock:
        if raw["TransactionID"] in online_engine.pending:
            raise HTTPException(
                409, f"Transaction {raw['TransactionID']} déjà scorée."
            )
        feats, info = online_engine.process(raw, commit=True)
    t1 = time.perf_counter()
    proba = float(_predict_proba(online_engine.vector(feats)[None, :])[0])
    t2 = time.perf_counter()
    decision_info = _decision(proba)
    _log_prediction("/predict/raw", proba, decision_info["decision"])

    return {
        "transaction_id": raw["TransactionID"],
        "fraud_probability": round(proba, 4),
        **decision_info,
        "thresholds": {
            "auto_approve_below": REVIEW_LOWER,
            "auto_block_at_or_above": BLOCK_UPPER,
        },
        "customer": {
            "previous_transactions": info["uid_history"],
            "has_fraud_label_history": not info["is_new_customer"],
            "past_fraud_rate": round(float(feats["uid_target_enc"]), 4),
        },
        "timing_ms": {
            "features": round((t1 - t0) * 1000, 2),
            "model": round((t2 - t1) * 1000, 2),
            "total": round((t2 - t0) * 1000, 2),
        },
    }


@app.post("/feedback", dependencies=[Depends(verify_api_key)])
@limiter.limit(STREAM_RATE_LIMIT)
def feedback(request: Request, fb: LabelFeedback):
    """
    Label confirmé d'une transaction déjà scorée par /predict/raw
    (ex : rétrofacturation reçue quelques jours plus tard). Met à jour
    les taux de fraude passés (target encoding) du client, de la carte,
    de l'adresse et du domaine e-mail. (protégé)
    """
    _require_engine()
    with engine_lock:
        ok = online_engine.add_label(fb.transaction_id, fb.is_fraud)
    if not ok:
        raise HTTPException(
            404, f"Transaction {fb.transaction_id} inconnue ou déjà labellisée."
        )
    return {"transaction_id": fb.transaction_id, "status": "label_recorded"}


@app.get("/online/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
def online_status(request: Request):
    """État du moteur temps réel : transactions vues, cartes, clients,
    labels en attente. (protégé)"""
    _require_engine()
    with engine_lock:
        return online_engine.state_summary()
