# -*- coding: utf-8 -*-
"""
Tests du mode temps réel : /predict/raw, /feedback, /online/status.

Prérequis : models/online_state.pkl et models/example_raw_transactions.json,
produits par `python scripts/build_online_state.py`. Sans eux, ces tests
sont ignorés (skip).
"""
import copy
import json
import os

os.environ["FRAUD_API_KEY"] = "test-key-pytest"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

HEADERS = {"x-api-key": "test-key-pytest"}
MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RAW_PATH = os.path.join(MODELS, "example_raw_transactions.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(MODELS, "online_state.pkl"))
         and os.path.exists(RAW_PATH)),
    reason="état en ligne absent : lancer scripts/build_online_state.py",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def raw_examples():
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)["examples"]


def _tx(example, new_id):
    tx = copy.deepcopy(example["transaction"])
    tx["TransactionID"] = new_id
    return tx


def test_realtime_enabled_in_health(client):
    assert client.get("/health").json()["realtime_enabled"] is True


def test_predict_raw_requires_api_key(client, raw_examples):
    r = client.post("/predict/raw", json={"transaction": _tx(raw_examples[0], 900001)})
    assert r.status_code == 401


def test_predict_raw_computes_features_online(client, raw_examples):
    """Une transaction brute est scorée ; la réponse donne la décision,
    l'historique du client et le temps de calcul."""
    r = client.post("/predict/raw", headers=HEADERS,
                    json={"transaction": _tx(raw_examples[2], 900002)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"auto_approve", "manual_review", "auto_block"}
    assert body["customer"]["previous_transactions"] >= 0
    assert 0.0 <= body["customer"]["past_fraud_rate"] <= 1.0
    assert body["timing_ms"]["total"] >= body["timing_ms"]["features"] >= 0


def test_predict_raw_updates_the_history(client, raw_examples):
    """La même carte, vue une 2e fois, a une transaction de plus dans
    son historique : l'état en ligne est mis à jour à chaque appel."""
    first = client.post("/predict/raw", headers=HEADERS,
                        json={"transaction": _tx(raw_examples[0], 900003)}).json()
    tx = _tx(raw_examples[0], 900004)
    second = client.post("/predict/raw", headers=HEADERS, json={"transaction": tx}).json()
    assert (second["customer"]["previous_transactions"]
            == first["customer"]["previous_transactions"] + 1)


def test_predict_raw_rejects_duplicate_transaction(client, raw_examples):
    tx = _tx(raw_examples[1], 900005)
    assert client.post("/predict/raw", headers=HEADERS, json={"transaction": tx}).status_code == 200
    r = client.post("/predict/raw", headers=HEADERS, json={"transaction": tx})
    assert r.status_code == 409


def test_predict_raw_rejects_missing_required_field(client, raw_examples):
    tx = _tx(raw_examples[0], 900006)
    del tx["card1"]
    r = client.post("/predict/raw", headers=HEADERS, json={"transaction": tx})
    assert r.status_code == 422


def test_predict_raw_rejects_negative_amount(client, raw_examples):
    tx = _tx(raw_examples[0], 900007)
    tx["TransactionAmt"] = -5
    r = client.post("/predict/raw", headers=HEADERS, json={"transaction": tx})
    assert r.status_code == 422


def test_feedback_raises_the_customer_fraud_rate(client, raw_examples):
    """Un label de fraude confirmé augmente le taux de fraude passé du
    client pour sa transaction suivante."""
    base = raw_examples[0]
    before = client.post("/predict/raw", headers=HEADERS,
                         json={"transaction": _tx(base, 900010)}).json()
    r = client.post("/feedback", headers=HEADERS,
                    json={"transaction_id": 900010, "is_fraud": 1})
    assert r.status_code == 200
    tx = _tx(base, 900011)
    after = client.post("/predict/raw", headers=HEADERS, json={"transaction": tx}).json()
    assert after["customer"]["past_fraud_rate"] > before["customer"]["past_fraud_rate"]
    assert after["customer"]["has_fraud_label_history"] is True


def test_feedback_unknown_or_repeated_is_rejected(client, raw_examples):
    assert client.post("/feedback", headers=HEADERS,
                       json={"transaction_id": 123456789, "is_fraud": 0}).status_code == 404
    client.post("/predict/raw", headers=HEADERS, json={"transaction": _tx(raw_examples[0], 900020)})
    assert client.post("/feedback", headers=HEADERS,
                       json={"transaction_id": 900020, "is_fraud": 0}).status_code == 200
    assert client.post("/feedback", headers=HEADERS,
                       json={"transaction_id": 900020, "is_fraud": 0}).status_code == 404


def test_online_status_counts_transactions(client, raw_examples):
    before = client.get("/online/status", headers=HEADERS).json()["transactions_seen"]
    client.post("/predict/raw", headers=HEADERS, json={"transaction": _tx(raw_examples[0], 900030)})
    after = client.get("/online/status", headers=HEADERS).json()["transactions_seen"]
    assert after == before + 1
