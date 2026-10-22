# Fraud Detection - Deployment (VS Code)

PFE project — Hamza Chaabene — TEK-UP University × Capgemini

## 🎬 Video demo

[▶️ Watch the demo (6 min 53, narrated in English)](docs/demo.mp4) — a real screen recording: the FastAPI API (Swagger), then the 5 dashboard tabs (Overview, Live stream, Check a transaction, Model comparison, Explore data).

## 📁 Structure

```
fraud-detection/
├── models/               ← produced by CELL 18 of the notebook (see below)
│   ├── catboost_model.cbm       ← CatBoost (standalone model + stacking base)
│   ├── stack_lgb_model.pkl      ← LightGBM base of the stacking
│   ├── stack_xgb_model.pkl      ← XGBoost base of the stacking
│   ├── stack_meta_model.pkl     ← meta-model (logistic regression)
│   ├── stacking_config.json     ← files + input order of the meta-model
│   ├── preprocessing.pkl        ← medians, encoders (feature completion)
│   ├── feature_names.json       ← 512 features expected by the API
│   ├── thresholds.json          ← thresholds + "production_model"
│   ├── dashboard_metrics.json   ← numbers shown by the dashboard (CELL 18)
│   ├── example_transactions.json← real test transactions (1 per decision, CELL 18)
│   ├── online_state.pkl         ← online history at the start of the test period (scripts/build_online_state.py)
│   ├── example_raw_transactions.json ← the 3 test transactions in RAW form
│   ├── online_parity.json / replay_report.json / feedback_experiment.json
│   │                            ← results of the real-time checks
│   ├── lgb_model.pkl / xgb_model.pkl / autoencoder.keras / scaler.pkl
│   │                            ← comparison models (not used by the API)
│   └── feature_names_lgb_xgb.json
├── api/
│   └── main.py           ← FastAPI (prediction backend)
├── realtime/
│   ├── engine.py          ← online feature engine (card / customer history)
│   └── scoring.py         ← model loading for the scripts
├── scripts/
│   ├── build_online_state.py   ← builds the online state + checks parity
│   ├── replay_stream.py        ← replays the test period through the API (latency)
│   ├── feedback_experiment.py  ← effect of confirmed labels (/feedback)
│   └── export_stream_sample.py ← data/stream_sample.csv.gz for the Live tab
├── experiments/
│   └── rf_isolation_forest.py  ← Random Forest and Isolation Forest (report, Section 4.5)
├── dashboard/
│   └── app.py             ← Streamlit (user interface)
├── data/stream_sample.csv.gz ← 3,000 raw test transactions (Live tab)
├── tests/                 ← pytest tests (API + real time)
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Step 1: Copy the models

Copy the content of the `models/` folder produced by CELL 18 of the
notebook into the `models/` folder at the root of this project.

### Production model

The model used is read from `thresholds.json` (key
`production_model`), and can be forced with the environment variable
`FRAUD_MODEL`:

| Value | Model | Test result (alerts ≤ 5%) |
|---|---|---|
| `stacking` | LightGBM + XGBoost + CatBoost → logistic regression | AUC 0.918 — net benefit $1,229,000 |
| `catboost` | CatBoost alone | AUC 0.907 — net benefit $1,175,880 |

The decision thresholds (`<model>_profit_optimized` → manual review,
`<model>` (F1) → block) are chosen by the notebook on the validation
set, for the model concerned.

```powershell
$env:FRAUD_MODEL="catboost"   # optional: force CatBoost alone
```

### Online history (real time)

The `/predict/raw` endpoint computes the 512 features of a **raw**
transaction itself, from a history of cards and customers. This
history is built once, from the notebook's raw data
(`train_transaction.csv` and `train_identity.csv`, in the parent
folder by default):

```bash
python scripts/build_online_state.py      # ~3 min -> models/online_state.pkl
```

The script replays train + validation (472,432 transactions), saves
the state on the first day of the test period, then replays the whole
test period transaction by transaction and checks that it reproduces
the notebook's results exactly (AUC 0.918, 2,524 frauds detected,
$1,229,000). Without this file, the API still starts, but
`/predict/raw` answers 503.

## 🔐 Step 2: Set the API key (security)

All endpoints except `/health` (`/predict/raw`, `/feedback`,
`/predict`, `/predict/batch`, `/model/info`, ...) are protected by an
API key. Set it before starting the server (replace it with a secret
value of your choice):

```powershell
# Windows PowerShell
$env:FRAUD_API_KEY="your-secret-key"
```
```bash
# Linux / Mac
export FRAUD_API_KEY="your-secret-key"
```

⚠️ If this variable is not set, the API still starts (to make
development easier) but prints a warning and uses an insecure default
key (`dev-only-change-me`) — never use it outside local development.

The Streamlit dashboard reads the same key from `.env` automatically
(server side: it is never sent to the browser), so there is nothing to
type. Another key can be used through "Use a different key" in the
sidebar. The API address can also be set in `.env` with
`FRAUD_API_URL`.

## 🚀 Step 3: Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Check: http://localhost:8000/health and
http://localhost:8000/docs (interactive Swagger documentation).

### Real-time endpoints

| Endpoint | Role |
|---|---|
| `POST /predict/raw` | scores a **raw** transaction (`{"transaction": {...CSV columns...}}`); updates the history |
| `POST /feedback` | label confirmed afterwards (`{"transaction_id": ..., "is_fraud": 0/1}`) |
| `GET /online/status` | size of the online history |

A transaction that was already scored is rejected (409). The rate
limit of these endpoints (`FRAUD_STREAM_RATE_LIMIT`, 6000/minute by
default) is that of a payment stream, not of a user.

Simulation of a real stream (API started from the saved state —
restart it before each replay):

```bash
python scripts/replay_stream.py --limit 5000                       # parity + latency
python scripts/feedback_experiment.py                              # effect of labels (7 / 30 days)
```

Measured results: 13 ms per transaction (median round trip), scores
identical to the notebook; with labels confirmed after 7 days, AUC
0.941 and +9.6% net benefit for the same review load.

## 🖥️ Step 4: Start the dashboard

In a second terminal:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501 and connects
automatically to the API at `http://localhost:8000` (can be changed in
the sidebar).

The **Live stream** tab sends the transactions of
`data/stream_sample.csv.gz` (start of the test period, in order) to
`/predict/raw`, one by one, and shows the decisions, the share of
flagged transactions and the response time.

All the numbers shown (AUC, net benefit, thresholds…) are read from
`models/dashboard_metrics.json`, written by CELL 18 of the notebook:
after a retraining, copying `models/` again is enough, without
changing `dashboard/app.py`. The theme is defined in
`.streamlit/config.toml` (start Streamlit from the project root).

## 🧪 Automated tests

A suite of 31 `pytest` tests checks the real behavior of the API
(authentication, input validation, 3-tier decision system,
explainability, real-time scoring and feedback) against the **real
models** — no mocks, so that the tests match the real behavior. The
real-time tests (`tests/test_realtime.py`) are skipped if
`models/online_state.pkl` has not been built yet.

```bash
pytest tests/ -v
```

⚠️ Prerequisite: the `models/` folder must be filled (see Step 1
above), otherwise the tests fail when loading the model, exactly like
the API itself.

## ℹ️ Two ways to call the API

- **`/predict/raw` (real time)**: the raw transaction; the API
  computes all the features from its online history.
- **`/predict`, `/predict/batch`, `/predict/explain`**: an already
  computed feature vector (used by the dashboard examples). Missing
  features are filled with the training values of
  `preprocessing.pkl`. Note: `card1`, `card2`, `addr1`,
  `P_emaildomain` and `uid_target_enc` are **target-encoded** values
  there (smoothed fraud rate between 0 and 1), not the raw
  identifiers. In `stacking` mode, `/predict/explain` returns the
  stacking probability but the SHAP values of the CatBoost base model.

Known limitation: the online history lives in the memory of a single
process and is not saved when it stops (a restart starts again from
`online_state.pkl`). A deployment on several instances would put it in
a shared store (e.g. Redis).
