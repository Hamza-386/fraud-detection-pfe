"""
Streamlit Dashboard - Financial Fraud Detection
PFE project - Hamza Chaabene - TEK-UP / Capgemini

Run from the project root (so .streamlit/config.toml is used):
    streamlit run dashboard/app.py
"""
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# Same .env as the API (project root): the API key is read on the
# server side and never sent to the browser.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ENV_API_KEY = os.getenv("FRAUD_API_KEY", "")
DEFAULT_API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ============================================
# Notebook results: models/dashboard_metrics.json
# Written by CELL 18 of the notebook - the single source of every
# number shown here (nothing is typed in by hand). All metrics are
# measured on the TEST set; every threshold was chosen on validation.
# ============================================
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_PATH = MODELS_DIR / "dashboard_metrics.json"
EXAMPLES_PATH = MODELS_DIR / "example_transactions.json"


@st.cache_data
def load_json(path: str, mtime: float) -> dict:
    """`mtime` refreshes the cache when the file is regenerated."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if not METRICS_PATH.exists():
    st.error(
        f"❌ File not found: `{METRICS_PATH}`. Run CELL 18 of the "
        "notebook, then copy its `models/` folder into this project."
    )
    st.stop()

METRICS = load_json(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
COSTS = METRICS["costs"]
N_TEST = METRICS["test_set"]["n_transactions"]
N_FRAUDS = METRICS["test_set"]["n_frauds"]
PROD = next(r for r in METRICS["business"] if r["role"] == "production")
ALT = next(r for r in METRICS["business"] if r["role"] == "alternative")
PROD_MODEL_ROW = next(m for m in METRICS["models"] if m["role"] == "production")
BOOT = METRICS["bootstrap_vs_alternative"]
THRESHOLDS = METRICS["decision_thresholds"]
PRODUCTION_MODEL = METRICS["production_label"]

# One colour per role, identical in every chart
ROLE_COLORS = {
    "production": "#3B82F6",   # blue (= theme primaryColor)
    "alternative": "#93C5FD",  # light blue
    "reference": "#64748B",    # grey
}
ROLE_LABELS = {
    "production": "In production",
    "alternative": "Alternative",
    "reference": "For comparison",
}

DECISIONS = {
    "auto_approve": {
        "title": "Approve", "icon": "✅", "color": "#22C55E",
        "sentence": "Low fraud risk — the transaction goes through automatically.",
    },
    "manual_review": {
        "title": "Review", "icon": "🟠", "color": "#F59E0B",
        "sentence": "Uncertain — an analyst should check this transaction "
                    "before it is accepted.",
    },
    "auto_block": {
        "title": "Block", "icon": "⛔", "color": "#EF4444",
        "sentence": "High fraud risk — the transaction is blocked automatically.",
    },
}

# Plain-English names for the model's input features
FEATURE_GLOSSARY = {
    "TransactionAmt": "Transaction amount",
    "uid_target_enc": "Customer's past fraud rate",
    "card1": "Card's past fraud rate",
    "card2": "Card issuer's past fraud rate",
    "addr1": "Billing address's past fraud rate",
    "P_emaildomain": "Email domain's past fraud rate",
    "Amt_cents": "Cents part of the amount",
    "D1": "Days since the card was first used",
    "D1_norm": "Card first-use date (normalised)",
    "card6": "Card type (credit / debit)",
    "hour": "Hour of the day",
    "C_high_mean": "Average of key counts",
    "C_high_sum": "Sum of key counts",
    "V_top_sum": "Sum of top risk signals",
    "uid_amt_mean": "Customer's average amount",
    "Amt_vs_uid_mean": "Amount vs customer's average",
}
# Short suffixes keep chart labels compact; explained under the chart
ANONYMISED_KINDS = {"V": "risk signal", "C": "count", "D": "time gap", "M": "match check"}


def describe_feature(name: str) -> str:
    if name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[name]
    if name[:1] in ANONYMISED_KINDS and name[1:].isdigit():
        return f"{name} · {ANONYMISED_KINDS[name[0]]}"
    return name


def usd(x: float) -> str:
    """Amount for markdown text: "\\$" stops Streamlit from reading the
    text between two "$" signs as a LaTeX formula."""
    return f"\\${x:,.0f}"


def usd_short(x: float) -> str:
    """Compact amount for st.metric values (not markdown)."""
    return f"${x / 1e6:.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}k"


# ============================================
# Example transactions for the "Check a transaction" tab
# ============================================
TYPICAL_RATE = 0.035   # overall fraud rate = value used for unknown history
SIMPLE_EXAMPLES = [
    {"name": "Typical purchase", "true_label": None,
     "features": {"TransactionAmt": 50.0, "uid_target_enc": 0.0}},
    {"name": "Suspicious purchase", "true_label": None,
     "features": {"TransactionAmt": 500.0, "uid_target_enc": 1.0,
                  "card1": 0.5, "card2": 0.5, "addr1": 0.5,
                  "P_emaildomain": 0.5}},
]
EXAMPLES = []
if EXAMPLES_PATH.exists():
    for ex in load_json(str(EXAMPLES_PATH), EXAMPLES_PATH.stat().st_mtime)["examples"]:
        outcome = "fraud" if ex["true_label"] == 1 else "legitimate"
        EXAMPLES.append({
            "name": f"Real transaction — {ex['name'].split()[0].lower()} "
                    f"(it was {outcome})",
            "true_label": ex["true_label"],
            "features": ex["features"],
        })
EXAMPLES += SIMPLE_EXAMPLES

# The five fields a user can adjust (everything else keeps the
# example's values, or typical training values)
EDITABLE_FIELDS = [
    ("TransactionAmt", "Transaction amount ($)", "amount"),
    ("uid_target_enc", "Customer's past fraud rate", "rate"),
    ("card1", "Card's past fraud rate", "rate"),
    ("P_emaildomain", "Email domain's past fraud rate", "rate"),
    ("addr1", "Billing address's past fraud rate", "rate"),
]

# ============================================
# Sidebar: connection settings + live status
# ============================================
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.caption("PFE project — Capgemini / TEK-UP")
st.sidebar.subheader("Connection")
api_url = st.sidebar.text_input("API address", value=DEFAULT_API_URL)
if ENV_API_KEY:
    # Never pass ENV_API_KEY as a widget value: it would be sent to the
    # browser. It stays on the server; the field below is only an override.
    st.sidebar.caption("🔒 API key loaded from the `.env` file")
    with st.sidebar.expander("Use a different key"):
        override_key = st.text_input("API key", value="", type="password")
    api_key = override_key or ENV_API_KEY
else:
    api_key = st.sidebar.text_input(
        "API key", value="", type="password",
        help="Or set FRAUD_API_KEY in the project's .env file to skip "
             "this step.",
    )
AUTH_HEADERS = {"X-API-Key": api_key} if api_key else {}


@st.cache_data(ttl=30, show_spinner=False)
def api_status(url: str, key: str) -> tuple:
    """(level, message) — cached 30 s so the API is not called on every click."""
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code != 200:
            return "error", f"API answered with code {r.status_code}"
        model = r.json().get("production_model", "?")
        r2 = requests.get(f"{url}/model/info",
                          headers={"X-API-Key": key} if key else {}, timeout=3)
        if r2.status_code == 200:
            return "ok", f"Connected — model: {model}"
        return "warning", "API reachable, but the API key is missing or wrong"
    except requests.exceptions.RequestException:
        return "error", "API not reachable — is it running?"


status_level, status_message = api_status(api_url, api_key)
{"ok": st.sidebar.success, "warning": st.sidebar.warning,
 "error": st.sidebar.error}[status_level](status_message)
if st.sidebar.button("🔄 Check again"):
    api_status.clear()
    st.rerun()


def call_api(endpoint: str, payload: dict, timeout: int = 15):
    """Returns (response_json, None) or (None, error message)."""
    try:
        r = requests.post(f"{api_url}{endpoint}", json=payload,
                          headers=AUTH_HEADERS, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Start it, then check the address in the sidebar."
    except requests.exceptions.Timeout:
        return None, "The API took too long to answer. Please try again."
    if r.status_code in (401, 403):
        return None, "The API key is missing or wrong (see the sidebar)."
    if r.status_code == 429:
        return None, ("Too many requests in one minute (the API limits them "
                      "for security). Wait a moment, then try again.")
    if r.status_code != 200:
        return None, f"API error {r.status_code}: {r.text}"
    return r.json(), None


# ============================================
# Header
# ============================================
st.title("🛡️ Fraud Detection Dashboard")
st.caption(f"IEEE-CIS card transactions · Model in production: **{PRODUCTION_MODEL}**")

tab_overview, tab_live, tab_check, tab_compare, tab_explore = st.tabs(
    ["🏠 Overview", "⚡ Live stream", "🔍 Check a transaction",
     "📊 Model comparison", "📁 Explore data"]
)

# ============================================
# TAB 1: Overview
# ============================================
with tab_overview:
    st.markdown(
        f"### On **{N_TEST:,}** recent transactions, the model caught "
        f"**{PROD['tp'] / N_FRAUDS:.0%}** of the frauds while flagging only "
        f"**{PROD['alert_rate']:.0%}** of the transactions."
    )
    st.caption(
        "These transactions were never used to train the model or to "
        "choose its settings, so the results reflect real-world use."
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1, st.container(border=True):
        st.metric(
            "Frauds caught", f"{PROD['tp'] / N_FRAUDS:.1%}",
            help=f"{PROD['tp']:,} of the {N_FRAUDS:,} frauds were flagged.",
        )
    with k2, st.container(border=True):
        st.metric(
            "Transactions flagged", f"{PROD['alert_rate']:.2%}",
            f"limit: {COSTS['max_alert_rate']:.0%}", delta_color="off",
            help="Share of transactions sent for review or blocked. The "
                 "review team can handle at most 5% of transactions.",
        )
    with k3, st.container(border=True):
        st.metric(
            "Estimated savings", usd_short(PROD["net_benefit"]),
            help=f"Each fraud caught saves {usd(COSTS['fraud_caught'])}; each "
                 f"false alarm costs {usd(COSTS['false_alert'])} of review time.",
        )
    with k4, st.container(border=True):
        st.metric(
            "Model quality (AUC)", f"{PROD_MODEL_ROW['AUC']:.3f}",
            help="How well the model ranks frauds above normal transactions: "
                 "0.5 = random guess, 1.0 = perfect.",
        )

    n_alerts = PROD["tp"] + PROD["fp"]
    st.markdown(
        f"Out of **{n_alerts:,}** flagged transactions, **{PROD['tp']:,}** "
        f"were real frauds and **{PROD['fp']:,}** were false alarms. "
        f"**{PROD['fn']:,}** frauds were missed."
    )

    st.markdown("#### How each transaction is decided")
    st.caption(
        "The model gives every transaction a fraud risk score between 0% "
        "and 100%. The score decides one of three actions:"
    )
    zones = [
        ("auto_approve", f"score below {THRESHOLDS['auto_approve_below']:.1%}"),
        ("manual_review", f"score from {THRESHOLDS['auto_approve_below']:.1%} "
                          f"to {THRESHOLDS['auto_block_at_or_above']:.1%}"),
        ("auto_block", f"score of {THRESHOLDS['auto_block_at_or_above']:.1%} "
                       f"or more"),
    ]
    for col, (code, rule) in zip(st.columns(3), zones):
        d = DECISIONS[code]
        with col, st.container(border=True):
            st.markdown(f"### {d['icon']} {d['title']}")
            st.markdown(f"**{rule}**")
            st.caption(d["sentence"])

    with st.expander("❓ How to use this dashboard", expanded=True):
        st.markdown(
            """
            1. **Connect** — automatic: the API key is read from the
               project's `.env` file. The status box in the sidebar turns
               green when the API is ready.
            2. **Live stream** — send real transactions to the API one by
               one, in the order they happened, and watch the decisions,
               the share of flagged transactions and the response time.
            3. **Check a transaction** — pick an example, adjust its details
               if you want, and click **Check transaction** to see the
               decision and why it was made.
            4. **Model comparison** — see why this model was chosen: it
               saves the most money for the same review workload.
            5. **Explore data** — upload a CSV extract of the dataset to see
               fraud rates and amounts.
            """
        )

# ============================================
# TAB: Live stream (real-time scoring of raw transactions)
# ============================================
STREAM_PATH = Path(__file__).resolve().parent.parent / "data" / "stream_sample.csv.gz"
FEEDBACK_OPTIONS = {
    "Not sent (history frozen, as in the notebook)": None,
    "Confirmed 30 days after the purchase": 30,
    "Confirmed 7 days after the purchase": 7,
}


@st.cache_data
def load_stream(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def raw_payload(row: pd.Series) -> dict:
    """One raw transaction as JSON (missing values -> null, no label)."""
    out = {}
    for k, v in row.items():
        if k == "isFraud":
            continue
        if isinstance(v, float) and v != v:
            out[k] = None
        else:
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def stream_time(dt: int) -> str:
    return f"day {dt // 86400}, {dt % 86400 // 3600:02d}:{dt % 3600 // 60:02d}"


with tab_live:
    st.subheader("Live stream")
    st.markdown(
        "Transactions arrive **one by one, in the order they happened**, as a "
        "payment system would send them. Each one is sent **raw** to the API, "
        "which computes all the model's inputs itself from the history of the "
        "card and of the customer, scores the transaction, then adds it to "
        "that history."
    )
    if not STREAM_PATH.exists():
        st.info("Stream file not found: run `python scripts/export_stream_sample.py`.")
    else:
        stream = load_stream(str(STREAM_PATH), STREAM_PATH.stat().st_mtime)
        live = st.session_state.setdefault(
            "live", {"pos": 0, "rows": [], "pending": [], "labels_sent": 0, "skipped": 0})

        c1, c2 = st.columns([1, 2])
        n_send = c1.select_slider("Transactions to send", [20, 50, 100, 200, 500], value=100)
        fb_label = c2.radio(
            "Confirmed fraud labels (chargebacks)", list(FEEDBACK_OPTIONS),
            help="In reality a fraud is confirmed days after the purchase. When "
                 "labels are sent back (/feedback), the customer's and the card's "
                 "past fraud rates are updated for their next purchases.",
        )
        delay_days = FEEDBACK_OPTIONS[fb_label]
        b1, b2, _ = st.columns([1, 1, 3])
        go_clicked = b1.button(f"▶ Send next {n_send}", type="primary",
                               disabled=live["pos"] >= len(stream))
        if b2.button("↺ Clear the view"):
            st.session_state["live"] = {"pos": 0, "rows": [], "pending": [],
                                        "labels_sent": 0, "skipped": 0}
            st.rerun()
        st.caption(
            f"{live['pos']:,} / {len(stream):,} transactions of the stream sent. "
            "The API keeps its history: restart it to replay the stream from the "
            "start (the first transactions of the test period, right after the "
            "saved history)."
        )

        kpi = st.empty()
        chart_box = st.empty()
        table_box = st.empty()

        def render():
            rows = live["rows"]
            if not rows:
                kpi.info("Press **Send** to start the stream.")
                return
            df_l = pd.DataFrame(rows)
            flagged = df_l["decision"] != "auto_approve"
            frauds = df_l["actual"] == 1
            lat = df_l["round_trip_ms"]
            with kpi.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Transactions scored", f"{len(df_l):,}")
                k2.metric("Flagged (review + block)", f"{flagged.mean():.1%}",
                          f"limit {COSTS['max_alert_rate']:.0%}", delta_color="off")
                k3.metric("Frauds caught", f"{int((flagged & frauds).sum())} / {int(frauds.sum())}",
                          help="Frauds of the stream sent for review or blocked. "
                               "The actual outcome is only known later; it is shown "
                               "here to evaluate the stream.")
                k4.metric("Response time (median)", f"{lat.median():.0f} ms",
                          f"95%: {lat.quantile(0.95):.0f} ms", delta_color="off",
                          help="Round trip dashboard -> API -> dashboard, including "
                               "the computation of the 512 features.")
                if live["labels_sent"] or live["skipped"]:
                    st.caption(f"Labels sent back: {live['labels_sent']:,} · "
                               f"already scored by the API (skipped): {live['skipped']:,}")
            fig = go.Figure()
            for code, info in DECISIONS.items():
                part = df_l[df_l["decision"] == code]
                fig.add_trace(go.Scatter(
                    x=part["n"], y=part["score"], mode="markers", name=info["title"],
                    marker={"color": info["color"], "size": 6},
                    hovertemplate="Transaction %{x}<br>Risk score %{y:.1%}<extra></extra>"))
            fr = df_l[frauds]
            fig.add_trace(go.Scatter(
                x=fr["n"], y=fr["score"], mode="markers", name="Actually fraud",
                marker={"symbol": "circle-open", "size": 12, "color": "#E2E8F0",
                        "line": {"width": 1.5}}, hoverinfo="skip"))
            for y, name in [(THRESHOLDS["auto_approve_below"], "review from"),
                            (THRESHOLDS["auto_block_at_or_above"], "block from")]:
                fig.add_hline(y=y, line_dash="dot", line_color="#94A3B8",
                              annotation_text=f"{name} {y:.1%}",
                              annotation_position="top left")
            fig.update_layout(
                height=330, margin={"t": 20, "b": 30, "l": 10, "r": 10},
                xaxis_title="Transactions, in order of arrival",
                yaxis={"title": "Fraud risk score", "type": "log",
                       "tickformat": ".1%",
                       "range": [math.log10(df_l["score"].min()) - 0.1, 0.02]},
                legend={"orientation": "h", "y": -0.25})
            st.session_state["live_chart_n"] = st.session_state.get("live_chart_n", 0) + 1
            chart_box.plotly_chart(fig, use_container_width=True,
                                   key=f"live_chart_{st.session_state['live_chart_n']}")
            last = df_l.tail(12).iloc[::-1]
            table_box.dataframe(
                pd.DataFrame({
                    "time": last["time"],
                    "amount ($)": last["amount"].map("{:,.2f}".format),
                    "product": last["product"],
                    "customer's past purchases": last["history"],
                    "risk score": last["score"].map("{:.1%}".format),
                    "decision": last["decision"].map(
                        lambda c: f"{DECISIONS[c]['icon']} {DECISIONS[c]['title']}"),
                    "actual (known later)": last["actual"].map({1: "fraud", 0: "legitimate"}),
                    "time (ms)": last["round_trip_ms"].map("{:.0f}".format),
                }),
                use_container_width=True, hide_index=True)

        if go_clicked:
            progress = st.progress(0.0)
            start = live["pos"]
            end = min(start + n_send, len(stream))
            session = requests.Session()
            error = None
            for i in range(start, end):
                row = stream.iloc[i]
                raw = raw_payload(row)
                now = raw["TransactionDT"]
                # labels whose confirmation date has passed are sent back first
                if delay_days is not None:
                    due = [p for p in live["pending"] if p[0] <= now]
                    live["pending"] = [p for p in live["pending"] if p[0] > now]
                    for _, tid, lab in due:
                        session.post(f"{api_url}/feedback", headers=AUTH_HEADERS,
                                     json={"transaction_id": tid, "is_fraud": lab}, timeout=10)
                        live["labels_sent"] += 1
                try:
                    t0 = time.perf_counter()
                    r = session.post(f"{api_url}/predict/raw", headers=AUTH_HEADERS,
                                     json={"transaction": raw}, timeout=10)
                    rtt = (time.perf_counter() - t0) * 1000
                except requests.exceptions.RequestException:
                    error = "Cannot reach the API. Start it, then check the address in the sidebar."
                    break
                live["pos"] = i + 1
                if r.status_code == 409:
                    live["skipped"] += 1
                    continue
                if r.status_code == 503:
                    error = ("The API's real-time engine is not loaded: run "
                             "`python scripts/build_online_state.py`, then restart the API.")
                    break
                if r.status_code != 200:
                    error = f"API error {r.status_code}: {r.text[:200]}"
                    break
                res = r.json()
                live["rows"].append({
                    "n": len(live["rows"]) + 1, "time": stream_time(now),
                    "amount": raw["TransactionAmt"], "product": raw.get("ProductCD"),
                    "history": res["customer"]["previous_transactions"],
                    "score": max(res["fraud_probability"], 1e-4),
                    "decision": res["decision"], "actual": int(row["isFraud"]),
                    "round_trip_ms": rtt,
                })
                if delay_days is not None:
                    live["pending"].append((now + delay_days * 86400,
                                            raw["TransactionID"], int(row["isFraud"])))
                progress.progress((i + 1 - start) / (end - start))
                if (i + 1) % 10 == 0:
                    render()
            progress.empty()
            live["error"] = error
            st.rerun()   # refresh the counter and the Send button
        if live.get("error"):
            st.error(f"❌ {live.pop('error')}")
        render()

# ============================================
# TAB 2: Check a transaction
# ============================================
with tab_check:
    st.subheader("Check a transaction")

    st.markdown("**Step 1 — Choose a starting example**")
    ex_index = st.radio(
        "Example", range(len(EXAMPLES)),
        format_func=lambda i: EXAMPLES[i]["name"],
        label_visibility="collapsed",
    )
    example = EXAMPLES[ex_index]
    if not EXAMPLES_PATH.exists():
        st.caption(
            "ℹ️ Real examples from the test set appear here after running "
            "CELL 18 of the notebook and copying `models/` into this project."
        )

    st.markdown("**Step 2 — Adjust the key details (optional)**")
    st.caption(
        "Past fraud rates come from the history of the customer, card, email "
        f"domain or address ({TYPICAL_RATE:.1%} ≈ typical). All other "
        "details keep the example's values."
    )
    edited, initial = {}, {}
    cols = st.columns(len(EDITABLE_FIELDS))
    for col, (field, label, kind) in zip(cols, EDITABLE_FIELDS):
        current = float(example["features"].get(
            field, 100.0 if kind == "amount" else TYPICAL_RATE))
        with col:
            # the key includes the example, so switching example resets values
            if kind == "amount":
                initial[field] = round(current, 2)
                edited[field] = st.number_input(
                    label, min_value=0.0, value=initial[field], step=10.0,
                    key=f"{field}_{ex_index}",
                )
            else:
                initial[field] = min(max(current, 0.0), 1.0)
                edited[field] = st.slider(
                    label, 0.0, 1.0, value=initial[field],
                    step=0.01, format="%.2f", key=f"{field}_{ex_index}",
                )
    # Only fields the user actually changed override the example, so an
    # untouched real example is scored exactly as in the notebook.
    overrides = {
        f: v for f, v in edited.items() if abs(v - initial[f]) > 1e-9
    }

    st.markdown("**Step 3 — Get the decision**")
    explain = st.checkbox("Also explain why (slower)", value=True)

    if st.button("🔍 Check transaction", type="primary"):
        features = {**example["features"], **overrides}
        endpoint = "/predict/explain" if explain else "/predict"
        res, error = call_api(endpoint, {"features": features})
        if error:
            st.error(f"❌ {error}")
        else:
            decision = res["decision"]
            d = DECISIONS[decision]
            proba = res["fraud_probability"]

            # Traffic light: the active decision is highlighted
            for col, code in zip(st.columns(3), DECISIONS):
                info = DECISIONS[code]
                active = code == decision
                with col:
                    st.markdown(
                        f"<div style='text-align:center;padding:14px;"
                        f"border-radius:10px;border:2px solid {info['color']};"
                        f"background:{info['color'] + '33' if active else 'transparent'};"
                        f"opacity:{1 if active else 0.35}'>"
                        f"<div style='font-size:28px'>{info['icon']}</div>"
                        f"<div style='font-size:20px;font-weight:700'>{info['title']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown(f"#### {d['icon']} {d['title']}: {d['sentence']}")
            t = res["thresholds"]
            st.markdown(
                f"Fraud risk score: **{proba:.1%}** "
                f"(approve below {t['auto_approve_below']:.1%}, "
                f"block from {t['auto_block_at_or_above']:.1%})."
            )
            if example["true_label"] is not None:
                truth = "**fraud**" if example["true_label"] == 1 else "**legitimate**"
                st.info(f"📌 In reality, this test transaction was {truth}.")

            if explain and "top_contributing_features" in res:
                st.markdown("#### Why this decision?")
                st.caption(
                    "The details that pushed the score up (red) or down "
                    "(green) the most. Explanation from the CatBoost part "
                    "of the model."
                )
                why = pd.DataFrame(res["top_contributing_features"])
                why["detail"] = why["feature"].map(describe_feature)
                why["effect_label"] = why["effect"].map({
                    "augmente_le_risque": "Raises the risk",
                    "diminue_le_risque": "Lowers the risk",
                })
                fig_why = px.bar(
                    why.sort_values("shap_value", key=abs),
                    x="shap_value", y="detail", orientation="h",
                    color="effect_label",
                    color_discrete_map={"Raises the risk": "#EF4444",
                                        "Lowers the risk": "#22C55E"},
                    labels={"shap_value": "Impact on the score",
                            "detail": "", "effect_label": ""},
                )
                fig_why.update_layout(
                    height=320, margin={"t": 10, "b": 10},
                    legend={"orientation": "h", "y": -0.2, "x": 0},
                )
                st.plotly_chart(fig_why, use_container_width=True)
                st.caption(
                    "V, C, D and M details are anonymised by the dataset "
                    "provider: V = risk signals, C = counts (e.g. addresses "
                    "linked to the card), D = time gaps in days, M = match "
                    "checks (e.g. name vs card)."
                )

    st.markdown("**Step 4 (optional) — See how one detail changes the risk**")
    st.caption(
        "Keeps the transaction above and tries 21 values of one detail. "
        "Uses one API call."
    )
    whatif_fields = {label: (field, kind) for field, label, kind in EDITABLE_FIELDS}
    whatif_label = st.selectbox("Detail to vary", list(whatif_fields), key="whatif_field")
    if st.button("📈 Show what-if chart"):
        field, kind = whatif_fields[whatif_label]
        base = {**example["features"], **overrides}
        current_value = float(edited[field])
        if kind == "amount":
            top = max(2000.0, current_value * 2)
            grid = [round(top * i / 20, 2) for i in range(21)]
        else:
            grid = [i / 20 for i in range(21)]
        res, error = call_api(
            "/predict/batch",
            {"transactions": [{**base, field: v} for v in grid]},
            timeout=30,
        )
        if error:
            st.error(f"❌ {error}")
        else:
            scores = [r["fraud_probability"] for r in res["results"]]
            decisions = [r["decision"] for r in res["results"]]
            t = res["thresholds"]
            lo, hi = t["auto_approve_below"], t["auto_block_at_or_above"]
            y_top = max(hi * 1.25, max(scores) * 1.15)

            fig_wi = go.Figure()
            # Coloured background = the three decision zones
            for code, (y0, y1) in zip(
                DECISIONS, [(0, lo), (lo, hi), (hi, max(y_top, hi))]
            ):
                info = DECISIONS[code]
                fig_wi.add_hrect(
                    y0=y0, y1=y1, fillcolor=info["color"], opacity=0.13,
                    line_width=0, annotation_text=f"{info['icon']} {info['title']}",
                    annotation_position="top left",
                )
            fig_wi.add_trace(go.Scatter(
                x=grid, y=scores, mode="lines+markers",
                # curve colour must contrast with the theme background
                line={"color": "#0F172A" if st.get_option("theme.base") == "light"
                      else "#E2E8F0", "width": 3},
                marker={"size": 6},
                hovertemplate=f"{whatif_label}: %{{x}}<br>Risk score: %{{y:.1%}}<extra></extra>",
                showlegend=False,
            ))
            fig_wi.add_vline(
                x=current_value, line_dash="dash", line_color="#3B82F6",
                annotation_text="this transaction",
                annotation_position="bottom left",
            )
            fig_wi.update_layout(
                xaxis={"title": whatif_label},
                yaxis={"title": "Fraud risk score", "tickformat": ".0%",
                       "range": [0, y_top]},
                height=380, margin={"l": 10, "r": 10, "t": 30, "b": 40},
            )
            st.plotly_chart(fig_wi, use_container_width=True,
                            config={"displayModeBar": False})

            # One plain sentence summarising the curve
            changes = [
                (grid[i], decisions[i]) for i in range(1, len(grid))
                if decisions[i] != decisions[i - 1]
            ]
            start = DECISIONS[decisions[0]]["title"]
            if not changes:
                st.info(
                    f"Whatever the {whatif_label.split(' (')[0].lower()}, "
                    f"the decision stays "
                    f"**{start}** (score from {min(scores):.1%} to {max(scores):.1%})."
                )
            else:
                steps = ", then ".join(
                    f"**{DECISIONS[d]['title']}** from about {v:g}" for v, d in changes
                )
                st.info(f"The decision starts at **{start}**, and becomes {steps}.")

    with st.expander("⚙️ Advanced: send raw features (JSON)"):
        st.caption(
            "For technical users: any feature names from `feature_names.json`. "
            "Missing features take typical training values."
        )
        json_input = st.text_area(
            "Features (JSON)", height=150,
            value=json.dumps({"TransactionAmt": 125.5, "uid_target_enc": 0.03}, indent=2),
        )
        if st.button("Send JSON"):
            try:
                feats = json.loads(json_input)
            except json.JSONDecodeError:
                st.error("❌ This is not valid JSON.")
            else:
                res, error = call_api("/predict", {"features": feats})
                if error:
                    st.error(f"❌ {error}")
                else:
                    d = DECISIONS[res["decision"]]
                    st.markdown(
                        f"{d['icon']} **{d['title']}** — fraud risk score "
                        f"**{res['fraud_probability']:.1%}**"
                    )

    with st.expander("📄 Check many transactions from a CSV file"):
        st.caption(
            "One row per transaction (up to 500), columns = feature names. "
            "Empty cells are filled by the API with typical training values. "
            "Text columns (e.g. a name or ID) are kept in the results but not "
            "sent to the model."
        )
        sample_df = pd.DataFrame(
            [{"example": ex["name"], **ex["features"]} for ex in EXAMPLES]
        )
        st.download_button(
            "📥 Download a sample CSV to try",
            sample_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_transactions.csv", mime="text/csv",
            help="The examples of Step 1, one per row. Upload it below.",
        )
        batch_file = st.file_uploader("CSV file", type=["csv"], key="batch_upload")
        if batch_file is not None and st.button("Check all transactions"):
            df_batch = pd.read_csv(batch_file)
            # Only numeric columns are model features; text columns are
            # kept for display (sending them would be rejected by the API).
            numeric = df_batch.select_dtypes(include="number")
            text_cols = [c for c in df_batch.columns if c not in numeric.columns]
            transactions = [
                {k: v for k, v in row.items() if pd.notna(v)}
                for row in numeric.to_dict(orient="records")
            ]
            res, error = call_api("/predict/batch", {"transactions": transactions}, timeout=30)
            if error:
                st.error(f"❌ {error}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Approve", res["n_auto_approve"])
                c2.metric("🟠 Review", res["n_manual_review"])
                c3.metric("⛔ Block", res["n_auto_block"])
                results_only = pd.DataFrame(res["results"]).drop(columns=["index"])
                results_only["decision"] = results_only["decision"].map(
                    lambda c: f"{DECISIONS[c]['icon']} {DECISIONS[c]['title']}")
                results_only = results_only.rename(columns={
                    "fraud_probability": "risk score", "risk_level": "risk level"})
                res_df = pd.concat([df_batch.reset_index(drop=True), results_only], axis=1)
                # On screen: identifying columns + results (a full feature
                # file has ~500 columns); the download keeps everything.
                shown = text_cols + [c for c in ("TransactionAmt",) if c in df_batch.columns]
                st.dataframe(
                    pd.concat([df_batch[shown].reset_index(drop=True), results_only], axis=1)
                    .style.format({"risk score": "{:.1%}", "TransactionAmt": "{:,.2f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.download_button(
                    "📥 Download results (CSV)",
                    res_df.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_check_results.csv", mime="text/csv",
                )

# ============================================
# TAB 3: Model comparison
# ============================================
def model_label(name: str, role: str) -> str:
    return {"production": f"{name} ★", "alternative": f"{name} (alt.)"}.get(role, name)


def row_style(row: pd.Series, roles: pd.Series) -> list:
    """Highlights the production row (text stays readable on a dark background)."""
    style = {
        "production": "background-color: rgba(59,130,246,0.28); font-weight: 600",
        "alternative": "background-color: rgba(147,197,253,0.12)",
    }.get(roles[row.name], "")
    return [style] * len(row)


def threshold_type_label(raw: str) -> str:
    return "Profit-optimised" if "profit" in raw.lower() else "F1-optimised"


def dot_plot(df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Dot (lollipop) chart: the axis can be zoomed without misleading
    the reader - unlike bars, which must start at 0 - so differences
    of 0.01 in AUC become visible.
    """
    df = df.sort_values(metric)
    lo = max(0.0, df[metric].min() - 0.03)
    hi = min(1.0, df[metric].max() + 0.04)
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[lo, r[metric]], y=[r["label"]] * 2, mode="lines",
            line={"color": ROLE_COLORS[r["role"]], "width": 2},
            hoverinfo="skip", showlegend=False,
        ))
    for role in ("production", "alternative", "reference"):
        sub = df[df["role"] == role]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[metric], y=sub["label"], mode="markers+text",
            name=ROLE_LABELS[role],
            marker={"color": ROLE_COLORS[role], "size": 13},
            text=[f"{v:.3f}" for v in sub[metric]],
            textposition="middle right",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis={"range": [lo, hi], "title": metric},
        yaxis={"categoryorder": "array", "categoryarray": list(df["label"])},
        height=38 * len(df) + 90,
        margin={"l": 10, "r": 10, "t": 40, "b": 40},
        legend={"orientation": "h", "y": 1.06, "x": 0},
    )
    return fig


with tab_compare:
    st.subheader("Why this model?")
    gain = PROD["net_benefit"] - ALT["net_benefit"]
    st.markdown(
        f"Every model was given the **same review capacity** (at most "
        f"{COSTS['max_alert_rate']:.0%} of transactions flagged). The "
        f"**{PRODUCTION_MODEL.split(' (')[0]}** model saves the most money: "
        f"**{usd(gain)} more** than the best single model."
    )

    bdf = pd.DataFrame(METRICS["business"])
    bdf["label"] = [model_label(n, r) for n, r in zip(bdf["name"], bdf["role"])]
    bdf = bdf.sort_values("net_benefit")
    fig_benefit = go.Figure(go.Bar(
        x=bdf["net_benefit"], y=bdf["label"], orientation="h",
        marker_color=[ROLE_COLORS[r] for r in bdf["role"]],
        # labels INSIDE the bars: never cut off, whatever the screen width
        text=[f"${v:,.0f}" for v in bdf["net_benefit"]],
        textposition="inside", insidetextanchor="end",
        textfont={"size": 14, "color": ["#0F172A" if r == "alternative" else "#FFFFFF"
                                        for r in bdf["role"]]},
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    ))
    fig_benefit.update_layout(
        title={"text": "Estimated savings on the test transactions", "x": 0},
        xaxis={"range": [0, bdf["net_benefit"].max() * 1.03], "tickprefix": "$"},
        height=300, margin={"l": 10, "r": 10, "t": 40, "b": 30},
        showlegend=False,
    )
    st.plotly_chart(fig_benefit, use_container_width=True)
    st.caption(
        f"Savings = frauds caught × {usd(COSTS['fraud_caught'])} − false "
        f"alarms × {usd(COSTS['false_alert'])}."
    )

    simple = pd.DataFrame({
        "Model": bdf["label"].to_numpy(),
        "Frauds caught": [f"{tp:,} of {N_FRAUDS:,} ({tp / N_FRAUDS:.0%})"
                          for tp in bdf["tp"]],
        "False alarms": [f"{fp:,}" for fp in bdf["fp"]],
        "Transactions flagged": [f"{a:.1%}" for a in bdf["alert_rate"]],
        "Savings": [f"${v:,.0f}" for v in bdf["net_benefit"]],
    }).iloc[::-1].reset_index(drop=True)
    roles_simple = bdf["role"].iloc[::-1].reset_index(drop=True)
    st.dataframe(
        simple.style.apply(row_style, roles=roles_simple, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.success(
        f"✅ The advantage is reliable: re-checking on {BOOT['n_resamples']} "
        f"random re-samples of the test data, the gain over CatBoost alone "
        f"stays between {usd(BOOT['ci_low'])} and {usd(BOOT['ci_high'])} "
        f"(95% range)."
    )

    with st.expander("🔬 Technical details (all models, AUC, F1…)"):
        mdf = pd.DataFrame(METRICS["models"])
        mdf["label"] = [model_label(n, r) for n, r in zip(mdf["name"], mdf["role"])]
        metric = st.selectbox(
            "Metric", ["AUC", "F1", "Precision", "Recall"],
            help="AUC does not depend on a threshold. F1 / Precision / "
                 "Recall are measured at each model's own threshold.",
        )
        st.plotly_chart(dot_plot(mdf, metric), use_container_width=True)

        mdf_sorted = mdf.sort_values("AUC", ascending=False).reset_index(drop=True)
        table_m = pd.DataFrame({
            "Model": mdf_sorted["label"],
            "Role": mdf_sorted["role"].map(ROLE_LABELS),
            "Threshold": mdf_sorted["threshold_type"].map(threshold_type_label),
            "AUC": mdf_sorted["AUC"],
            "Recall": mdf_sorted["Recall"],
            "Precision": mdf_sorted["Precision"],
            "F1": mdf_sorted["F1"],
        })
        st.dataframe(
            table_m.style
            .apply(row_style, roles=mdf_sorted["role"], axis=1)
            .format({c: "{:.4f}" for c in ("AUC", "Recall", "Precision", "F1")}),
            use_container_width=True, hide_index=True,
        )
        st.markdown(
            f"""
            - **Stacking V2** combines LightGBM, XGBoost and CatBoost with a
              logistic regression fitted on the first half of the validation
              data; its thresholds are chosen on the second half. Best AUC
              ({PROD_MODEL_ROW['AUC']:.3f}) and best savings.
            - **CatBoost_TE** (CatBoost alone, same features) is a simpler
              alternative: the API switches to it with `FRAUD_MODEL=catboost`.
            - Models without a suffix (LightGBM / XGBoost / CatBoost) do not
              use target encoding; they are comparison baselines only.
            - F1 / Precision / Recall are only comparable between models with
              the same type of threshold.
            """
        )

# ============================================
# TAB 4: Explore data (uploaded CSV)
# ============================================
with tab_explore:
    st.subheader("Explore the data")
    st.markdown(
        "Upload an extract of `train_transaction.csv` (or any CSV with at "
        "least `TransactionAmt`, and `isFraud` if available)."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="explore_upload")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", f"{len(df):,}")
        if "isFraud" in df.columns:
            col2.metric("Fraud rate", f"{df['isFraud'].mean():.2%}")
            col3.metric("Frauds", f"{int(df['isFraud'].sum()):,}")

        if "TransactionAmt" in df.columns:
            st.markdown("#### Transaction amounts")
            plot_df = df.copy()
            color = None
            if "isFraud" in plot_df.columns:
                plot_df["Type"] = plot_df["isFraud"].map({0: "Legitimate", 1: "Fraud"})
                color = "Type"
            fig = px.histogram(
                plot_df, x="TransactionAmt", color=color,
                nbins=60, marginal="box", barmode="overlay", opacity=0.7,
                labels={"TransactionAmt": "Amount ($)"},
                color_discrete_map={"Legitimate": "#3B82F6", "Fraud": "#EF4444"},
            )
            st.plotly_chart(fig, use_container_width=True)

        if "ProductCD" in df.columns and "isFraud" in df.columns:
            st.markdown("#### Fraud rate by product type")
            rate_by_product = (
                df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False) * 100
            ).reset_index(name="Fraud rate (%)")
            fig2 = px.bar(rate_by_product, x="ProductCD", y="Fraud rate (%)",
                          labels={"ProductCD": "Product type"})
            fig2.update_traces(marker_color="#3B82F6")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👆 Waiting for a CSV file. No data is loaded automatically.")
