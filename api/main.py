"""
FastAPI - Real-Time Fraud Detection Service (secured)
PFE project - Hamza Chaabene - TEK-UP / Capgemini

Loads the production model produced by CELL 18 of the notebook and
exposes protected REST endpoints for prediction.

Production model ("production_model" key of thresholds.json, can be
overridden with the FRAUD_MODEL environment variable):
- "stacking": LightGBM + XGBoost + CatBoost (full models of CELL 13B)
  combined by a logistic regression
  (stack_*_model.pkl + catboost_model.cbm, see stacking_config.json)
- "catboost": CatBoost alone (catboost_model.cbm)

Features missing from a request are filled with the training values of
preprocessing.pkl (training median, or global fraud rate for the
target-encoded columns).

Security:
- API key authentication (X-API-Key header), read from a .env file
  (see .env.example) with python-dotenv — never hard-coded in the
  source code.
- Rate limiting per IP
- Strict input validation (finite values, non-empty dict)

Three-tier decision (instead of a binary threshold):
- proba <  review_threshold (profit-optimized)      -> auto_approve
- review_threshold <= proba < block_threshold (F1)  -> manual_review
- proba >= block_threshold                          -> auto_block

Run with:
    uvicorn api.main:app --reload --port 8000

API key configuration (recommended):
    1. Copy .env.example to .env: cp .env.example .env
    2. Edit .env and replace the value of FRAUD_API_KEY
    3. The .env file is never versioned (see .gitignore)
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
# Logging (traceability of predictions)
# ============================================
# Every prediction is logged for audit. A simple log file is enough
# for this prototype; in production, this would typically be sent to
# a centralized system (ELK, CloudWatch, etc.).
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
    """Logs a single prediction for later audit."""
    logger.info(
        json.dumps({
            "endpoint": endpoint,
            "fraud_probability": round(proba, 4),
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    )

# Loads the variables of the .env file (if it exists) into the process
# environment, BEFORE they are read below with os.getenv(). Does not
# override a variable already set by the system.
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
# Real time: online state (history per card / customer) built by
# scripts/build_online_state.py. Missing -> /predict/raw unavailable.
ONLINE_STATE_PATH = os.getenv(
    "FRAUD_ONLINE_STATE", os.path.join(MODELS_DIR, "online_state.pkl")
)
# /predict/raw is called for EVERY payment by the payment server
# (a single IP): its limit is that of a stream, not of a human.
STREAM_RATE_LIMIT = os.getenv("FRAUD_STREAM_RATE_LIMIT", "6000/minute")

limiter = Limiter(key_func=get_remote_address)

# Global state loaded at startup
# `model` = CatBoost: production model in "catboost" mode, base model
# (and the model explained by SHAP) in "stacking" mode.
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
default_values: Dict[str, float] = {}       # fill values for missing features
# Online feature engine (realtime.engine.OnlineFeatureEngine); its state
# is shared between requests, hence the lock.
online_engine = None
engine_lock = threading.Lock()

# Bounds of the 3-tier decision system (computed at startup from
# thresholds.json, with a fallback if the keys are missing)
REVIEW_LOWER: float = 0.3   # below this threshold: auto_approve
BLOCK_UPPER: float = 0.7    # at or above: auto_block


def _load_default_values(path: str) -> Dict[str, float]:
    """
    Values used to fill the features missing from a request, the same
    as in the notebook:
    - training median (CELL 10) for the imputed columns;
    - global fraud rate of the encoders (CELL 13B) for the
      target-encoded columns and uid_target_enc — exactly the value an
      unknown key receives.
    Other columns (never missing in training) -> 0.0.
    """
    if not os.path.exists(path):
        print(
            f"⚠️  {path} not found — missing features will be "
            f"filled with 0.0."
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
    print(f"✅ Fill values loaded ({len(defaults)} features)")
    return defaults


def _load_stacking() -> None:
    """Loads the LightGBM / XGBoost base models and the meta-model."""
    global stack_base_models, stack_meta_model, stack_input_order

    config_path = os.path.join(MODELS_DIR, "stacking_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"❌ {config_path} not found: 'stacking' mode needs the "
            f"files produced by CELL 18 of the notebook."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # CatBoost (already loaded in `model`) is the 3rd base model
    if config["base_models"].get("catboost") != "catboost_model.cbm":
        raise RuntimeError("❌ stacking_config.json: unexpected CatBoost model.")

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
            f"{stack_input_order} does not match the models {expected}."
        )
    print(
        f"✅ Stacking loaded: {' + '.join(stack_input_order)} "
        f"-> {type(stack_meta_model).__name__}"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces the deprecated
    @app.on_event("startup")). The code before `yield` runs at startup;
    after `yield`, at shutdown (nothing to do here, but the extension
    point exists if needed later, e.g. closing a database connection).
    """
    global model, feature_names, thresholds, explainer
    global REVIEW_LOWER, BLOCK_UPPER
    global production_model, stack_base_models, stack_meta_model
    global stack_input_order, default_values

    if API_KEY == DEFAULT_API_KEY:
        print(
            "⚠️  WARNING: FRAUD_API_KEY is not set — using an insecure "
            "default key. Set the FRAUD_API_KEY environment variable "
            "before any use outside local development."
        )

    model_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    features_path = os.path.join(MODELS_DIR, "feature_names.json")
    thresholds_path = os.path.join(MODELS_DIR, "thresholds.json")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"❌ Model not found: {model_path}\n"
            f"Check that the 'models/' folder (produced by CELL 18 of "
            f"the notebook) is at the root of the project."
        )

    model = CatBoostClassifier()
    model.load_model(model_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    print(f"✅ CatBoost model loaded ({len(feature_names)} features)")

    default_values = _load_default_values(
        os.path.join(MODELS_DIR, "preprocessing.pkl")
    )

    production_model = os.getenv(
        "FRAUD_MODEL", thresholds.get("production_model", "catboost")
    ).strip().lower()
    if production_model not in SUPPORTED_MODELS:
        raise RuntimeError(
            f"❌ Unknown production model: '{production_model}' "
            f"(accepted values: {', '.join(SUPPORTED_MODELS)})"
        )
    if production_model == "stacking":
        _load_stacking()

    print(f"✅ Production model: {production_model}")
    print(
        "✅ Production threshold (profit-optimized): "
        f"{thresholds.get(f'{production_model}_profit_optimized', 0.5)}"
    )

    # REVIEW_LOWER = profit-optimized threshold (start of the doubt zone)
    # BLOCK_UPPER  = F1-optimized threshold, historically higher
    #                (high confidence -> automatic block)
    # Both are chosen on validation by the notebook, for the
    # production model itself ("stacking" or "catboost").
    REVIEW_LOWER = thresholds.get(f"{production_model}_profit_optimized", 0.3)
    BLOCK_UPPER = thresholds.get(production_model, 0.7)
    if BLOCK_UPPER <= REVIEW_LOWER:
        # Safeguard: if the saved thresholds are inconsistent, keep a
        # minimal gap so that the manual review zone really exists.
        BLOCK_UPPER = min(REVIEW_LOWER + 0.3, 0.95)
    print(
        f"✅ 3-tier decision: auto_approve < {REVIEW_LOWER:.2f} "
        f"<= manual_review < {BLOCK_UPPER:.2f} <= auto_block"
    )

    explainer = shap.TreeExplainer(model)
    print("✅ SHAP explainer ready (TreeExplainer on CatBoost)")

    global online_engine
    if os.path.exists(ONLINE_STATE_PATH):
        with open(ONLINE_STATE_PATH, "rb") as f:
            online_engine = pickle.load(f)
        if list(online_engine.feature_names) != list(feature_names):
            raise RuntimeError(
                "❌ online_state.pkl does not match feature_names.json: "
                "run scripts/build_online_state.py again."
            )
        print(f"✅ Real-time engine loaded: {online_engine.state_summary()}")
    else:
        online_engine = None
        print(
            f"ℹ️  {ONLINE_STATE_PATH} missing: /predict/raw disabled "
            f"(run scripts/build_online_state.py)."
        )

    yield  # The API is ready and serves requests here

    # --- Shutdown code ---
    print("👋 Fraud Detection API stopping")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud detection API (stacking of LightGBM + "
        "XGBoost + CatBoost, or CatBoost alone) "
        "- PFE project (endpoints protected by an API key)"
    ),
    version="1.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
def root():
    """Points human visitors to the Swagger documentation."""
    return {"message": "Fraud Detection API", "docs": "/docs"}


@app.exception_handler(Exception)
async def _hide_internal_errors(request: Request, exc: Exception):
    """
    Avoids returning a raw Python stack trace (which could expose
    internal details) for any unexpected error not already handled by
    an explicit HTTPException.
    """
    from fastapi.responses import JSONResponse
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ============================================
# Security: API key authentication
# ============================================
def verify_api_key(x_api_key: str = Header(default=None)):
    """FastAPI dependency checking the X-API-Key header."""
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing 'X-API-Key' header.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return True


# ============================================
# Pydantic schemas (with strict validation)
# ============================================
class Transaction(BaseModel):
    """
    An already preprocessed transaction: dictionary {feature_name: value}.
    Features missing with respect to feature_names.json are filled
    automatically with the training values (preprocessing.pkl: median,
    or global fraud rate for the target-encoded columns).

    Important: card1, card2, addr1, P_emaildomain and uid_target_enc
    are TARGET-ENCODED values (smoothed fraud rate, between 0 and 1),
    not the raw identifiers.

    Note: this schema expects a feature vector that is already computed
    (e.g. by an upstream feature engineering job). To score a raw
    transaction, with the features computed from the online history,
    use /predict/raw.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of preprocessed features",
        examples=[{"TransactionAmt": 125.50, "card1_freq": 340}],
    )

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v):
        if not v:
            raise ValueError("The feature dictionary is empty.")
        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(
                    f"Non-finite value (NaN/inf) for feature "
                    f"'{key}'."
                )
        return v


class BatchTransactions(BaseModel):
    transactions: List[Dict[str, float]]

    @field_validator("transactions")
    @classmethod
    def _validate_batch(cls, v):
        if not v:
            raise ValueError("The transaction list is empty.")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch too large ({len(v)} transactions) — "
                f"limit: {MAX_BATCH_SIZE}."
            )
        for i, feats in enumerate(v):
            for key, val in feats.items():
                if not math.isfinite(val):
                    raise ValueError(
                        f"Transaction {i}: non-finite value for "
                        f"'{key}'."
                    )
        return v


RAW_REQUIRED = ("TransactionID", "TransactionDT", "TransactionAmt", "card1")


class RawTransaction(BaseModel):
    """
    A RAW transaction, as it arrives from the payment system: columns
    of train_transaction.csv + train_identity.csv (without isFraud).
    Absent or null columns are treated as missing, as in the notebook.
    All the features (UID, card and customer history, target
    encoding, ...) are computed online by the API.
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
                raise ValueError(f"Missing required field: '{key}'.")
        for key in ("TransactionDT", "TransactionAmt", "card1"):
            val = v[key]
            if isinstance(val, str) or not math.isfinite(val):
                raise ValueError(f"'{key}' must be a finite number.")
        if v["TransactionAmt"] < 0 or v["TransactionDT"] < 0:
            raise ValueError("TransactionAmt and TransactionDT must be >= 0.")
        for key, val in v.items():
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError(f"Non-finite value for '{key}'.")
        return v


class LabelFeedback(BaseModel):
    """Label confirmed afterwards (e.g. a chargeback), which updates the
    fraud history of the customer, the card, ..."""
    transaction_id: Union[int, str]
    is_fraud: int = Field(..., ge=0, le=1)


# ============================================
# Helper functions
# ============================================
def _complete_row(feats: Dict[str, float]) -> Dict[str, float]:
    """Full vector: the given value, otherwise the training value."""
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
    """Fraud probability of a base model through its native booster
    (LightGBM: booster_.predict; XGBoost: inplace_predict), without the
    scikit-learn wrapper. Same result as predict_proba[:, 1]."""
    if hasattr(base_model, "booster_"):          # LGBMClassifier
        return base_model.booster_.predict(A)
    if hasattr(base_model, "get_booster"):       # XGBClassifier
        return base_model.get_booster().inplace_predict(A)
    return base_model.predict_proba(A)[:, 1]


def _predict_proba(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Fraud probability of the production model.
    - "catboost": output of CatBoost.
    - "stacking": probabilities of the 3 base models, combined by the
      meta-model in the order of stacking_config.json.
    """
    # numpy array rather than DataFrame: same probabilities, but the
    # pandas validation of LightGBM / XGBoost cost ~30 ms per call.
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
    Determines the three-tier operational action, instead of a simple
    binary is_fraud true/false threshold:
    - auto_approve  : low risk, the transaction is processed normally
    - manual_review : uncertainty zone, to be checked by an analyst
                      before the final decision
    - auto_block    : high confidence of fraud, immediate block
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
    Checks that the service and the model are operational.
    Deliberately PUBLIC (no key required) to allow simple external
    monitoring (e.g. load balancer, uptime checker).
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
    """Returns the metadata of the loaded model. (protected)"""
    if model is None:
        raise HTTPException(503, "Model not loaded")
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
    """Predicts the fraud probability of ONE transaction. (protected)"""
    if model is None:
        raise HTTPException(503, "Model not loaded")

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
        description="Number of most contributing features to return",
    ),
):
    """
    Predicts the fraud probability AND explains the decision with SHAP
    values (contribution of each feature to this specific transaction).
    More expensive than /predict — meant for the manual review of
    alerts, not for bulk scoring.

    In "stacking" mode, the probability and the decision are those of
    the stacking, but the SHAP values explain the CatBoost base model
    (the meta-model has no exact SHAP explanation).
    """
    if model is None or explainer is None:
        raise HTTPException(503, "Model or explainer not loaded")

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
            "effect": "increases_risk" if val > 0 else "decreases_risk",
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
    """Predicts the fraud probability of a LIST of transactions.
    (protected, limited to MAX_BATCH_SIZE transactions per call)"""
    if model is None:
        raise HTTPException(503, "Model not loaded")

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
# Real time: raw transactions
# ============================================
def _require_engine():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if online_engine is None:
        raise HTTPException(
            503,
            "Real-time engine not loaded (models/online_state.pkl missing: "
            "run scripts/build_online_state.py).",
        )


@app.post("/predict/raw", dependencies=[Depends(verify_api_key)])
@limiter.limit(STREAM_RATE_LIMIT)
def predict_raw(request: Request, tx: RawTransaction):
    """
    Scores a RAW transaction when it arrives. (protected)

    The API computes the 512 features itself from its online state
    (history of the card, of the reconstructed customer, of the email
    domain, ...), exactly as the notebook does, then updates this state
    with the transaction. No label is needed at this point: labels
    arrive later through /feedback.
    """
    _require_engine()
    t0 = time.perf_counter()
    raw = tx.transaction
    with engine_lock:
        if raw["TransactionID"] in online_engine.pending:
            raise HTTPException(
                409, f"Transaction {raw['TransactionID']} already scored."
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
    Confirmed label of a transaction already scored by /predict/raw
    (e.g. a chargeback received a few days later). Updates the past
    fraud rates (target encoding) of the customer, the card, the
    address and the email domain. (protected)
    """
    _require_engine()
    with engine_lock:
        ok = online_engine.add_label(fb.transaction_id, fb.is_fraud)
    if not ok:
        raise HTTPException(
            404, f"Transaction {fb.transaction_id} unknown or already labeled."
        )
    return {"transaction_id": fb.transaction_id, "status": "label_recorded"}


@app.get("/online/status", dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
def online_status(request: Request):
    """State of the real-time engine: transactions seen, cards,
    customers, pending labels. (protected)"""
    _require_engine()
    with engine_lock:
        return online_engine.state_summary()
