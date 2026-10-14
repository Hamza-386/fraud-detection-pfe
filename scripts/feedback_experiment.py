"""
Effect of label feedback on the test period.

In production a fraud is confirmed some days after the purchase (a
chargeback). With /feedback, the confirmed labels update the past fraud
rates (target encoding) of the customer, card, address and e-mail
domain, which the notebook kept frozen at the end of training.

This script replays the whole test period through the online engine,
starting from models/online_state.pkl, once without feedback (= the
notebook's evaluation) and once per label delay. The production
thresholds are kept as chosen on validation.

Usage: python scripts/feedback_experiment.py [--delays 30 7]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.engine import iter_records  # noqa: E402
from realtime.scoring import StackingScorer  # noqa: E402


def run(state_path, records, labels, delay_days, scorer, feature_names):
    with open(state_path, "rb") as f:
        engine = pickle.load(f)
    delay = None if delay_days is None else delay_days * 86400
    waiting = deque()
    X = np.empty((len(records), len(feature_names)))
    n_labels = 0
    for i, rec in enumerate(records):
        now = rec["TransactionDT"]
        while waiting and waiting[0][0] <= now:
            _, tid, y = waiting.popleft()
            n_labels += engine.add_label(tid, y)
        feats, _ = engine.process(rec)
        X[i] = engine.vector(feats)
        if delay is not None:
            waiting.append((now + delay, rec["TransactionID"], int(labels[i])))
    return scorer.predict_all(pd.DataFrame(X, columns=feature_names))["stacking"], n_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, ".."))
    ap.add_argument("--delays", type=float, nargs="*", default=[30, 7])
    args = ap.parse_args()
    mdir = os.path.join(ROOT, "models")

    tr = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    idt = pd.read_csv(os.path.join(args.data_dir, "train_identity.csv"))
    df = tr.merge(idt, on="TransactionID", how="left")
    del tr, idt
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(df)
    test = df.iloc[int(n * 0.70) + int(n * 0.10):]
    del df
    labels = test["isFraud"].to_numpy()
    records = list(iter_records(test.drop(columns=["isFraud"])))

    with open(os.path.join(mdir, "thresholds.json"), encoding="utf-8") as f:
        thr = json.load(f)
    t_review, t_block = thr["stacking_profit_optimized"], thr["stacking"]
    scorer = StackingScorer(mdir)
    results = []
    for delay in [None] + list(args.delays):
        p, n_labels = run(os.path.join(mdir, "online_state.pkl"), records, labels,
                          delay, scorer, scorer.feature_names)
        flag = p >= t_review
        tp = int((flag & (labels == 1)).sum())
        fp = int((flag & (labels == 0)).sum())
        r = {"label_delay_days": delay, "labels_received": int(n_labels),
             "auc": round(float(roc_auc_score(labels, p)), 4),
             "alert_rate": round(float(flag.mean()), 4),
             "frauds_caught": tp, "false_alarms": fp,
             "recall": round(tp / int(labels.sum()), 4),
             "net_benefit": tp * 500 - fp * 10,
             "auto_block": int((p >= t_block).sum())}
        # Same review capacity as without feedback: the k highest scores,
        # k = number of alerts of the frozen-history run.
        if delay is None:
            k = int(flag.sum())
        top = np.argsort(-p, kind="stable")[:k]
        tp_k = int(labels[top].sum())
        r["equal_capacity"] = {"alerts": k, "frauds_caught": tp_k,
                               "false_alarms": k - tp_k,
                               "net_benefit": tp_k * 500 - (k - tp_k) * 10}
        results.append(r)
        print(json.dumps(r))
    with open(os.path.join(mdir, "feedback_experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"thresholds": {"review": t_review, "block": t_block},
                   "n_test": len(labels), "n_frauds": int(labels.sum()),
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
