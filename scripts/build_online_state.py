"""
Builds the online state and checks training/serving parity.

1. Loads the raw IEEE-CIS data exactly as the notebook does (CELL 2:
   transaction + identity merge, then a stable chronological sort).
2. Replays the training and validation periods through the online
   engine (state updates only) and saves the state as it stands on the
   first day of the test period: models/online_state.pkl. The API
   starts from this state.
3. Replays the test period transaction by transaction, computing the
   features online, then scores them with the production models and
   compares with the notebook:
     - the 3 reference test transactions of example_transactions.json
       (full 512-feature vectors, built offline by the notebook);
     - the test metrics of dashboard_metrics.json (AUC, frauds caught,
       false alarms at the production threshold).

Usage (from the fraud-detection folder):
    python scripts/build_online_state.py [--data-dir ..]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from realtime.engine import OnlineFeatureEngine, iter_records  # noqa: E402
from realtime.scoring import StackingScorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, ".."))
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()
    mdir = args.models_dir

    t0 = time.time()
    tr = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    idt = pd.read_csv(os.path.join(args.data_dir, "train_identity.csv"))
    df = tr.merge(idt, on="TransactionID", how="left")
    del tr, idt
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.10)
    y = df["isFraud"].to_numpy()
    df = df.drop(columns=["isFraud"])
    print(f"Data loaded: {n:,} transactions ({time.time() - t0:.0f}s); "
          f"test = rows {val_end:,}..{n - 1:,}")

    preprocessing = joblib.load(os.path.join(mdir, "preprocessing.pkl"))
    with open(os.path.join(mdir, "feature_names.json"), encoding="utf-8") as f:
        feature_names = json.load(f)
    engine = OnlineFeatureEngine(preprocessing, feature_names)

    # ---- warm-up: train + validation history (no labels used)
    t0 = time.time()
    for rec in iter_records(df.iloc[:val_end]):
        engine.observe(rec)
    print(f"History replayed: {engine.state_summary()} ({time.time() - t0:.0f}s)")
    state_path = os.path.join(mdir, "online_state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"State saved: {state_path} ({os.path.getsize(state_path) / 1e6:.0f} MB)")

    # ---- test period, online, one transaction at a time
    n_test = n - val_end
    X = np.empty((n_test, len(feature_names)), dtype=np.float64)
    t0 = time.time()
    lat = []
    for i, rec in enumerate(iter_records(df.iloc[val_end:])):
        s = time.perf_counter()
        feats, _ = engine.process(rec)
        lat.append(time.perf_counter() - s)
        X[i] = engine.vector(feats)
    lat_ms = np.array(lat) * 1000
    print(f"Test replayed online: {n_test:,} transactions ({time.time() - t0:.0f}s); "
          f"feature time per transaction: p50 {np.percentile(lat_ms, 50):.2f} ms, "
          f"p95 {np.percentile(lat_ms, 95):.2f} ms, p99 {np.percentile(lat_ms, 99):.2f} ms")
    y_test = y[val_end:]

    # ---- parity 1: the 3 reference vectors built offline by the notebook
    with open(os.path.join(mdir, "example_transactions.json"), encoding="utf-8") as f:
        examples = json.load(f)["examples"]
    print("\nParity with the notebook's reference test transactions:")
    worst_all = 0.0
    for ex in examples:
        ref = np.array([ex["features"][c] for c in feature_names], dtype=np.float64)
        got = X[ex["test_row"]]
        diff = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
        worst = int(np.argmax(diff))
        worst_all = max(worst_all, diff[worst])
        bad = [(feature_names[j], ref[j], got[j]) for j in np.where(diff > 1e-5)[0]]
        print(f"  {ex['name']:18s} label={ex['true_label']} max rel. diff={diff[worst]:.2e} "
              f"({feature_names[worst]}); features differing: {len(bad)}")
        for name, r, g in bad[:15]:
            print(f"      {name}: notebook={r!r} online={g!r}")

    # ---- parity 2: model scores and test metrics
    scorer = StackingScorer(mdir)
    Xdf = pd.DataFrame(X, columns=feature_names)
    probas = scorer.predict_all(Xdf)
    for ex in examples:
        print(f"  {ex['name']:18s} notebook score={ex['score']:.6f} "
              f"online score={probas['stacking'][ex['test_row']]:.6f}")

    with open(os.path.join(mdir, "dashboard_metrics.json"), encoding="utf-8") as f:
        dm = json.load(f)
    with open(os.path.join(mdir, "thresholds.json"), encoding="utf-8") as f:
        thr = json.load(f)
    ref_auc = {m["name"]: m["AUC"] for m in dm["models"]}
    ref_bus = {b["name"]: b for b in dm["business"]}
    print("\nTest metrics, notebook vs online features:")
    report = {}
    for name, key, tkey in [("Stacking_V2", "stacking", "stacking_profit_optimized"),
                            ("CatBoost_TE", "catboost", "catboost_profit_optimized")]:
        p = probas[key]
        t = thr[tkey]
        flag = p >= t
        tp = int((flag & (y_test == 1)).sum())
        fp = int((flag & (y_test == 0)).sum())
        auc = roc_auc_score(y_test, p)
        b = ref_bus[name]
        print(f"  {name:12s} AUC {ref_auc[name]:.6f} -> {auc:.6f} | "
              f"frauds caught {b['tp']} -> {tp} | false alarms {b['fp']} -> {fp} | "
              f"net benefit {b['net_benefit']:,} -> {tp * 500 - fp * 10:,}")
        report[name] = {"auc_notebook": ref_auc[name], "auc_online": auc,
                        "tp_notebook": b["tp"], "tp_online": tp,
                        "fp_notebook": b["fp"], "fp_online": fp}

    out = {
        "n_test": n_test,
        "feature_latency_ms": {q: float(np.percentile(lat_ms, int(q[1:])))
                               for q in ("p50", "p95", "p99")},
        "reference_vectors_max_rel_diff": float(worst_all),
        "metrics": report,
        "state": engine.state_summary(),
    }
    with open(os.path.join(mdir, "online_parity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Raw form of the 3 reference transactions (for the tests and the
    # dashboard): what the payment system would send to /predict/raw.
    raw_examples = []
    for ex in examples:
        rec = df.iloc[val_end + ex["test_row"]].to_dict()
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else
                   v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
        raw_examples.append({
            "name": ex["name"], "expected_decision": ex["expected_decision"],
            "true_label": ex["true_label"], "test_row": ex["test_row"],
            "score_online_replay": float(probas["stacking"][ex["test_row"]]),
            "transaction": rec,
        })
    with open(os.path.join(mdir, "example_raw_transactions.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "scripts/build_online_state.py",
                   "note": "scores obtained when the online history is the one "
                           "replayed up to this transaction",
                   "examples": raw_examples}, f, indent=1)
    np.save(os.path.join(mdir, "online_test_scores.npy"), probas["stacking"])
    print(f"\nReport written: {os.path.join(mdir, 'online_parity.json')}")


if __name__ == "__main__":
    main()
