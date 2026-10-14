"""
Real-time simulation: replays the test period through the running API,
one raw transaction at a time, in chronological order, as a payment
server would send them.

  * each transaction is sent to POST /predict/raw (raw columns only,
    without isFraud): the API computes the features online;
  * optionally, its label is sent to POST /feedback once the simulated
    clock has moved `--feedback-delay-days` past it (a chargeback is
    confirmed some days after the purchase);
  * the round-trip latency and the API's own timing are measured, and
    the scores are compared with the offline parity run
    (models/online_test_scores.npy) when no feedback is sent.

The API must start from the state at the beginning of the test period
(models/online_state.pkl, built by scripts/build_online_state.py):
restart it before each replay.

Usage (API running on localhost:8000, key in .env):
    python scripts/replay_stream.py --limit 5000
    python scripts/replay_stream.py --limit 5000 --feedback-delay-days 7
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(rec: dict) -> dict:
    """JSON-safe raw transaction: NaN -> null, numpy -> Python types."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, ".."))
    ap.add_argument("--api", default=os.getenv("FRAUD_API_URL", "http://localhost:8000"))
    ap.add_argument("--limit", type=int, default=5000,
                    help="number of test transactions to replay (0 = all)")
    ap.add_argument("--feedback-delay-days", type=float, default=None,
                    help="send each label this many days after the purchase "
                         "(default: no feedback, frozen history = parity mode)")
    args = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    headers = {"X-API-Key": os.getenv("FRAUD_API_KEY", "")}

    tr = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    idt = pd.read_csv(os.path.join(args.data_dir, "train_identity.csv"))
    df = tr.merge(idt, on="TransactionID", how="left")
    del tr, idt
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(df)
    val_end = int(n * 0.70) + int(n * 0.10)
    test = df.iloc[val_end:] if args.limit == 0 else df.iloc[val_end:val_end + args.limit]
    labels = test["isFraud"].to_numpy()
    records = test.drop(columns=["isFraud"]).to_dict("records")
    del df

    session = requests.Session()
    status = session.get(f"{args.api}/online/status", headers=headers, timeout=10)
    status.raise_for_status()
    print(f"API state before replay: {status.json()}")

    delay = None if args.feedback_delay_days is None else args.feedback_delay_days * 86400
    waiting = deque()           # (time label becomes known, TransactionID, label)
    scores, rtt, server_total, server_feat, decisions = [], [], [], [], []
    n_feedback = 0
    t_start = time.time()
    for i, rec in enumerate(records):
        raw = clean(rec)
        now = raw["TransactionDT"]
        while delay is not None and waiting and waiting[0][0] <= now:
            _, tid, lab = waiting.popleft()
            r = session.post(f"{args.api}/feedback", headers=headers,
                             json={"transaction_id": tid, "is_fraud": int(lab)}, timeout=10)
            r.raise_for_status()
            n_feedback += 1
        s = time.perf_counter()
        r = session.post(f"{args.api}/predict/raw", headers=headers,
                         json={"transaction": raw}, timeout=10)
        rtt.append((time.perf_counter() - s) * 1000)
        if r.status_code != 200:
            sys.exit(f"Transaction {raw['TransactionID']}: HTTP {r.status_code} {r.text[:300]}")
        out = r.json()
        scores.append(out["fraud_probability"])
        decisions.append(out["decision"])
        server_total.append(out["timing_ms"]["total"])
        server_feat.append(out["timing_ms"]["features"])
        if delay is not None:
            waiting.append((now + delay, raw["TransactionID"], labels[i]))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} transactions | {time.time() - t_start:.0f}s | "
                  f"p50 round trip {np.percentile(rtt, 50):.1f} ms")
    elapsed = time.time() - t_start

    scores = np.array(scores)
    rtt = np.array(rtt)
    pct = lambda a, q: round(float(np.percentile(a, q)), 2)  # noqa: E731
    report = {
        "transactions": len(records),
        "feedback_delay_days": args.feedback_delay_days,
        "labels_sent": n_feedback,
        "throughput_tx_per_s": round(len(records) / elapsed, 1),
        "latency_ms": {
            "round_trip": {"p50": pct(rtt, 50), "p95": pct(rtt, 95), "p99": pct(rtt, 99)},
            "server_total": {"p50": pct(server_total, 50), "p95": pct(server_total, 95)},
            "server_features": {"p50": pct(server_feat, 50), "p95": pct(server_feat, 95)},
        },
        "decisions": {d: decisions.count(d) for d in ("auto_approve", "manual_review", "auto_block")},
        "frauds": int(labels.sum()),
        "auc": round(float(roc_auc_score(labels, scores)), 4) if 0 < labels.sum() < len(labels) else None,
    }
    flagged = np.array([d != "auto_approve" for d in decisions])
    report["frauds_caught"] = int((flagged & (labels == 1)).sum())
    report["false_alarms"] = int((flagged & (labels == 0)).sum())

    parity_path = os.path.join(ROOT, "models", "online_test_scores.npy")
    if args.feedback_delay_days is None and os.path.exists(parity_path):
        offline = np.load(parity_path)[:len(scores)]
        report["max_abs_diff_vs_offline_replay"] = float(np.max(np.abs(np.round(offline, 4) - scores)))

    print(json.dumps(report, indent=2))
    name = "replay_report.json" if delay is None else f"replay_report_feedback_{args.feedback_delay_days:g}d.json"
    with open(os.path.join(ROOT, "models", name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
