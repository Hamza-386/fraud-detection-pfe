# -*- coding: utf-8 -*-
"""
Tests automatisés de l'API de détection de fraude.
Projet PFE - Hamza Chaabene - TEK-UP / Capgemini

Prérequis : le dossier models/ (téléchargé depuis Kaggle) doit être
présent à la racine du projet, exactement comme pour lancer l'API
normalement — ces tests utilisent le VRAI modèle CatBoost, pas un
mock, afin de valider le comportement réel de bout en bout.

Lancer avec (depuis la racine du projet) :
    pytest tests/ -v
"""
import os

# IMPORTANT : la clé de test doit être définie AVANT d'importer
# api.main, car API_KEY y est lue une seule fois au chargement du
# module (os.getenv au niveau module, pas dans une fonction).
os.environ["FRAUD_API_KEY"] = "test-key-pytest"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

HEADERS = {"x-api-key": "test-key-pytest"}


@pytest.fixture(scope="module")
def client():
    """
    TestClient utilisé comme context manager pour déclencher les
    évènements startup/shutdown de FastAPI (chargement réel du
    modèle, de l'explainer SHAP, etc.).
    """
    with TestClient(app) as c:
        yield c


# ============================================
# /health — endpoint public
# ============================================
def test_health_is_public_and_ok(client):
    """/health ne doit exiger aucune clé API et confirmer le modèle chargé."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features_expected"] > 0


# ============================================
# Sécurité : authentification par clé API
# ============================================
def test_predict_without_api_key_is_rejected(client):
    r = client.post("/predict", json={"features": {"TransactionAmt": 50}})
    assert r.status_code == 401


def test_predict_with_wrong_api_key_is_rejected(client):
    r = client.post(
        "/predict",
        json={"features": {"TransactionAmt": 50}},
        headers={"x-api-key": "mauvaise-cle"},
    )
    assert r.status_code == 403


def test_model_info_requires_api_key(client):
    r = client.get("/model/info")
    assert r.status_code == 401


# ============================================
# Validation des entrées
# ============================================
def test_predict_rejects_empty_features(client):
    r = client.post("/predict", json={"features": {}}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_missing_body(client):
    r = client.post("/predict", json={}, headers=HEADERS)
    assert r.status_code == 422


def test_batch_rejects_empty_list(client):
    r = client.post(
        "/predict/batch", json={"transactions": []}, headers=HEADERS
    )
    assert r.status_code == 422


def test_batch_rejects_oversized_list(client):
    """Le lot ne doit pas dépasser MAX_BATCH_SIZE (500) transactions."""
    too_many = [{"TransactionAmt": 10.0} for _ in range(501)]
    r = client.post(
        "/predict/batch",
        json={"transactions": too_many},
        headers=HEADERS,
    )
    assert r.status_code == 422


# ============================================
# /model/info — cohérence des seuils
# ============================================
def test_model_info_thresholds_are_consistent(client):
    r = client.get("/model/info", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    lower = body["decision_thresholds"]["auto_approve_below"]
    upper = body["decision_thresholds"]["auto_block_at_or_above"]
    # Le seuil de blocage doit toujours être strictement supérieur
    # au seuil d'approbation automatique (garde-fou du système à
    # 3 niveaux).
    assert 0.0 <= lower < upper <= 1.0


# ============================================
# /predict — comportement du système à 3 niveaux
# ============================================
def test_predict_low_risk_transaction_is_auto_approved(client):
    """
    uid_target_enc=0.0 (aucun historique de fraude connu) doit
    produire une décision auto_approve, conformément à la
    validation manuelle effectuée précédemment.
    """
    r = client.post(
        "/predict",
        json={"features": {"TransactionAmt": 50, "uid_target_enc": 0.0}},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] == "auto_approve"
    assert body["risk_level"] == "low"


def test_predict_risk_increases_with_uid_fraud_history(client):
    """
    uid_target_enc élevé (historique de fraude du client) doit
    augmenter la probabilité de fraude.

    (Remplace l'ancien test « uid_target_enc=0.9 -> auto_block » :
    il dépendait des features manquantes complétées par 0.0 ; elles
    sont maintenant complétées par les valeurs d'entraînement, et une
    seule feature modifiée ne suffit pas à atteindre le blocage.)
    """
    def proba(uid_te):
        r = client.post(
            "/predict",
            json={"features": {"TransactionAmt": 500, "uid_target_enc": uid_te}},
            headers=HEADERS,
        )
        assert r.status_code == 200
        return r.json()["fraud_probability"]

    assert proba(0.9) > proba(0.0)


def test_decision_levels_match_thresholds(client):
    """Les 3 niveaux suivent exactement les seuils chargés."""
    from api import main

    lower, upper = main.REVIEW_LOWER, main.BLOCK_UPPER
    assert main._decision(max(lower - 1e-6, 0.0))["decision"] == "auto_approve"
    assert main._decision(lower)["decision"] == "manual_review"
    assert main._decision(upper)["decision"] == "auto_block"
    assert main._decision(upper)["risk_level"] == "high"


def test_predict_decision_is_consistent_with_returned_thresholds(client):
    """La décision renvoyée correspond à la probabilité et aux seuils renvoyés."""
    for feats in (
        {"TransactionAmt": 50, "uid_target_enc": 0.0},
        {"TransactionAmt": 500, "uid_target_enc": 1.0, "card1": 0.5,
         "card2": 0.5, "addr1": 0.5, "P_emaildomain": 0.5},
    ):
        body = client.post(
            "/predict", json={"features": feats}, headers=HEADERS
        ).json()
        p = body["fraud_probability"]
        t = body["thresholds"]
        if p >= t["auto_block_at_or_above"]:
            expected = "auto_block"
        elif p >= t["auto_approve_below"]:
            expected = "manual_review"
        else:
            expected = "auto_approve"
        # tolérance d'arrondi (probabilité arrondie à 4 décimales)
        if min(abs(p - t["auto_block_at_or_above"]),
               abs(p - t["auto_approve_below"])) > 1e-4:
            assert body["decision"] == expected


# ============================================
# Complétion des features & modèle de production
# ============================================
def test_missing_features_are_filled_with_training_values(client):
    """
    Les features absentes prennent la valeur d'entraînement
    (preprocessing.pkl), pas 0.0 : uid_target_enc absent = taux de
    fraude global (~3.5 %), comme une clé inconnue dans le notebook.
    """
    from api import main

    X = main._build_feature_vector({"TransactionAmt": 100.0})
    assert X.loc[0, "TransactionAmt"] == 100.0
    global_rate = main.default_values["uid_target_enc"]
    assert 0.0 < global_rate < 0.1
    assert X.loc[0, "uid_target_enc"] == pytest.approx(global_rate)


def test_model_info_reports_production_model(client):
    body = client.get("/model/info", headers=HEADERS).json()
    assert body["production_model"] in ("stacking", "catboost")
    assert body["explained_model"] == "catboost"
    if body["production_model"] == "stacking":
        assert body["stacking"]["meta_input_order"] == ["lgb", "xgb", "catboost"]
        assert len(body["stacking"]["meta_coefficients"]) == 3


def test_stacking_combines_the_three_base_models(client):
    """En mode stacking : proba = méta-modèle(P_lgb, P_xgb, P_catboost)."""
    import numpy as np
    from api import main

    if main.production_model != "stacking":
        pytest.skip("API en mode catboost")
    X = main._build_feature_vector({"TransactionAmt": 300.0, "uid_target_enc": 0.5})
    base = {
        "lgb": main.stack_base_models["lgb"].predict_proba(X)[:, 1],
        "xgb": main.stack_base_models["xgb"].predict_proba(X)[:, 1],
        "catboost": main.model.predict_proba(X)[:, 1],
    }
    meta_in = np.column_stack([base[n] for n in main.stack_input_order])
    expected = main.stack_meta_model.predict_proba(meta_in)[0, 1]
    assert main._predict_proba(X)[0] == pytest.approx(expected)


def test_predict_response_contains_expected_fields(client):
    r = client.post(
        "/predict",
        json={"features": {"TransactionAmt": 100}},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    for field in (
        "fraud_probability", "decision", "risk_level", "thresholds",
    ):
        assert field in body
    assert body["decision"] in (
        "auto_approve", "manual_review", "auto_block",
    )


def test_predict_ignores_unknown_feature_names(client):
    """
    Une feature inconnue (absente de feature_names.json) ne doit pas
    faire planter l'API — elle doit simplement être ignorée lors de
    la construction du vecteur.
    """
    r = client.post(
        "/predict",
        json={"features": {"colonne_qui_nexiste_pas": 42.0}},
        headers=HEADERS,
    )
    assert r.status_code == 200


# ============================================
# /predict/batch
# ============================================
def test_predict_batch_returns_consistent_counts(client):
    r = client.post(
        "/predict/batch",
        json={
            "transactions": [
                {"TransactionAmt": 50, "uid_target_enc": 0.0},
                {"TransactionAmt": 500, "uid_target_enc": 0.9},
            ]
        },
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_transactions"] == 2
    assert len(body["results"]) == 2
    assert (
        body["n_auto_approve"]
        + body["n_manual_review"]
        + body["n_auto_block"]
        == 2
    )


# ============================================
# /predict/explain
# ============================================
def test_predict_explain_returns_top_features(client):
    r = client.post(
        "/predict/explain?top_n=3",
        json={"features": {"TransactionAmt": 100, "uid_target_enc": 0.5}},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert "top_contributing_features" in body
    assert len(body["top_contributing_features"]) <= 3
    # mêmes champs de décision que /predict (utilisés par le dashboard)
    for field in ("fraud_probability", "decision", "risk_level", "thresholds"):
        assert field in body
    for feat in body["top_contributing_features"]:
        assert "feature" in feat
        assert "shap_value" in feat
        assert feat["effect"] in ("augmente_le_risque", "diminue_le_risque")


def test_predict_explain_top_n_out_of_range_is_rejected(client):
    r = client.post(
        "/predict/explain?top_n=999",
        json={"features": {"TransactionAmt": 100}},
        headers=HEADERS,
    )
    assert r.status_code == 422
