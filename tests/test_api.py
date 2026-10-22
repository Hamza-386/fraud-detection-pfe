# -*- coding: utf-8 -*-
"""
Automated tests of the fraud detection API.
PFE project - Hamza Chaabene - TEK-UP / Capgemini

Prerequisite: the models/ folder (produced by CELL 18 of the notebook)
must be at the root of the project, exactly as for running the API
normally — these tests use the REAL CatBoost model, not a mock, to
check the real end-to-end behavior.

Run with (from the project root):
    pytest tests/ -v
"""
import os

# IMPORTANT: the test key must be set BEFORE importing api.main,
# because API_KEY is read there only once, when the module is loaded
# (os.getenv at module level, not inside a function).
os.environ["FRAUD_API_KEY"] = "test-key-pytest"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

HEADERS = {"x-api-key": "test-key-pytest"}


@pytest.fixture(scope="module")
def client():
    """
    TestClient used as a context manager to trigger the FastAPI
    startup/shutdown events (real loading of the model, of the SHAP
    explainer, etc.).
    """
    with TestClient(app) as c:
        yield c


# ============================================
# /health — public endpoint
# ============================================
def test_health_is_public_and_ok(client):
    """/health must not require an API key and must confirm the model is loaded."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features_expected"] > 0


# ============================================
# Security: API key authentication
# ============================================
def test_predict_without_api_key_is_rejected(client):
    r = client.post("/predict", json={"features": {"TransactionAmt": 50}})
    assert r.status_code == 401


def test_predict_with_wrong_api_key_is_rejected(client):
    r = client.post(
        "/predict",
        json={"features": {"TransactionAmt": 50}},
        headers={"x-api-key": "wrong-key"},
    )
    assert r.status_code == 403


def test_model_info_requires_api_key(client):
    r = client.get("/model/info")
    assert r.status_code == 401


# ============================================
# Input validation
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
    """A batch must not exceed MAX_BATCH_SIZE (500) transactions."""
    too_many = [{"TransactionAmt": 10.0} for _ in range(501)]
    r = client.post(
        "/predict/batch",
        json={"transactions": too_many},
        headers=HEADERS,
    )
    assert r.status_code == 422


# ============================================
# /model/info — consistency of the thresholds
# ============================================
def test_model_info_thresholds_are_consistent(client):
    r = client.get("/model/info", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    lower = body["decision_thresholds"]["auto_approve_below"]
    upper = body["decision_thresholds"]["auto_block_at_or_above"]
    # The block threshold must always be strictly above the automatic
    # approval threshold (safeguard of the 3-tier system).
    assert 0.0 <= lower < upper <= 1.0


# ============================================
# /predict — behavior of the 3-tier system
# ============================================
def test_predict_low_risk_transaction_is_auto_approved(client):
    """
    uid_target_enc=0.0 (no known fraud history) must give an
    auto_approve decision, as checked manually before.
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
    A high uid_target_enc (customer fraud history) must raise the
    fraud probability.

    (Replaces the old "uid_target_enc=0.9 -> auto_block" test: it
    depended on missing features being filled with 0.0; they are now
    filled with the training values, and changing a single feature is
    not enough to reach the block level.)
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
    """The 3 levels follow the loaded thresholds exactly."""
    from api import main

    lower, upper = main.REVIEW_LOWER, main.BLOCK_UPPER
    assert main._decision(max(lower - 1e-6, 0.0))["decision"] == "auto_approve"
    assert main._decision(lower)["decision"] == "manual_review"
    assert main._decision(upper)["decision"] == "auto_block"
    assert main._decision(upper)["risk_level"] == "high"


def test_predict_decision_is_consistent_with_returned_thresholds(client):
    """The returned decision matches the returned probability and thresholds."""
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
        # rounding tolerance (probability rounded to 4 decimals)
        if min(abs(p - t["auto_block_at_or_above"]),
               abs(p - t["auto_approve_below"])) > 1e-4:
            assert body["decision"] == expected


# ============================================
# Feature completion & production model
# ============================================
def test_missing_features_are_filled_with_training_values(client):
    """
    Missing features take the training value (preprocessing.pkl), not
    0.0: a missing uid_target_enc = global fraud rate (~3.5%), like an
    unknown key in the notebook.
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
    """In stacking mode: proba = meta-model(P_lgb, P_xgb, P_catboost)."""
    import numpy as np
    from api import main

    if main.production_model != "stacking":
        pytest.skip("API in catboost mode")
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
    An unknown feature (absent from feature_names.json) must not crash
    the API — it must simply be ignored when the vector is built.
    """
    r = client.post(
        "/predict",
        json={"features": {"column_that_does_not_exist": 42.0}},
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
    # same decision fields as /predict (used by the dashboard)
    for field in ("fraud_probability", "decision", "risk_level", "thresholds"):
        assert field in body
    for feat in body["top_contributing_features"]:
        assert "feature" in feat
        assert "shap_value" in feat
        assert feat["effect"] in ("increases_risk", "decreases_risk")


def test_predict_explain_top_n_out_of_range_is_rejected(client):
    r = client.post(
        "/predict/explain?top_n=999",
        json={"features": {"TransactionAmt": 100}},
        headers=HEADERS,
    )
    assert r.status_code == 422
