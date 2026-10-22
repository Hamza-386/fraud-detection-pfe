"""Random Forest and Isolation Forest under the notebook's protocol.

Rebuilds X_train / X_val / X_test by executing the notebook's own cells
(CELL 2, 7, 7B, 8, 8B, 8C, 9, 10), then:
  - Random Forest (class-weighted), configuration chosen on validation AUC;
  - Isolation Forest (unsupervised), configuration chosen on validation AUC;
  - F1 threshold and profit threshold (<= 5% alerts) chosen on validation,
    applied once to test. Net benefit = TP*500 - FP*10.
Run from this folder: python rf_isolation_forest.py. The notebook's CELL 2
reads the raw CSVs from its own data paths. Results: rf_isolation_forest_results.json.
"""
import json, time, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)

from pathlib import Path
NB = Path(__file__).resolve().parent.parent / "notebook.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
CELLS = [3, 21, 23, 25, 27, 28, 30, 32, 34]

ns = {}
exec("import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings, os, json, joblib\n"
     "from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder\n"
     "from sklearn.metrics import *\n", ns)
ns["plt"].show = lambda *a, **k: None
t0 = time.time()
for i in CELLS:
    src = "".join(nb["cells"][i]["source"])
    print(f"\n>>> running notebook cell {i}", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)
    if i == 3:
        ns.pop("df_original", None); ns.pop("train_transaction", None); ns.pop("train_identity", None)
print(f"\nFeatures rebuilt in {time.time()-t0:.0f}s", flush=True)

X_train, X_val, X_test = ns["X_train"], ns["X_val"], ns["X_test"]
y_train, y_val, y_test = (np.asarray(ns[k]) for k in ("y_train", "y_val", "y_test"))
print("shapes", X_train.shape, X_val.shape, X_test.shape,
      "NaN left:", int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum()))
assert X_train.shape[1] == 511 and len(X_test) == 118108 and y_test.sum() == 4064
# sklearn forests need finite values
fill = X_train.median()
X_train, X_val, X_test = (X.fillna(fill).astype(np.float32) for X in (X_train, X_val, X_test))

COST_FN, COST_FP, CAP = 500, 10, 0.05

def evaluate(name, s_val, s_test, prob=True):
    """Thresholds chosen on validation, applied once to test."""
    if prob:
        grid_f1 = np.arange(0.1, 0.9, 0.01)
        grid_profit = np.arange(0.01, 0.99, 0.01)
    else:  # unbounded anomaly scores: grid of validation quantiles
        grid_f1 = grid_profit = np.quantile(s_val, np.linspace(0.5, 0.999, 500))
    f1s = [f1_score(y_val, s_val > t) for t in grid_f1]
    t_f1 = grid_f1[int(np.argmax(f1s))]
    best, t_p = -np.inf, None
    for t in grid_profit:
        pred = s_val > t
        if pred.mean() > CAP:
            continue
        p = ((pred & (y_val == 1)).sum() * COST_FN - (pred & (y_val == 0)).sum() * COST_FP)
        if p > best:
            best, t_p = p, t
    out = {"model": name,
           "val_auc": roc_auc_score(y_val, s_val),
           "test_auc": roc_auc_score(y_test, s_test),
           "test_ap": average_precision_score(y_test, s_test)}
    pf = s_test > t_f1
    out.update(f1_threshold=float(t_f1), precision=precision_score(y_test, pf),
               recall=recall_score(y_test, pf), f1=f1_score(y_test, pf))
    pp = s_test > t_p
    tp, fp = int((pp & (y_test == 1)).sum()), int((pp & (y_test == 0)).sum())
    out.update(profit_threshold=float(t_p), profit_alert_rate=float(pp.mean()),
               profit_tp=tp, profit_fp=fp, net_benefit=tp * COST_FN - fp * COST_FP)
    print(json.dumps(out, indent=1), flush=True)
    return out

results = []

# ---------- Random Forest: small search on validation AUC ----------
best_rf = None
for leaf in (20, 100):
    t = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=leaf,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    print(f"RF min_samples_leaf={leaf}: val AUC {auc:.4f} ({time.time()-t:.0f}s)", flush=True)
    if best_rf is None or auc > best_rf[0]:
        best_rf = (auc, leaf, rf, time.time() - t)
    else:
        del rf
_, leaf, rf, rf_time = best_rf
r = evaluate(f"RandomForest (leaf={leaf})", rf.predict_proba(X_val)[:, 1], rf.predict_proba(X_test)[:, 1])
r["train_time_s"] = rf_time
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
r["top_features"] = imp.head(10).round(4).to_dict()
results.append(r)
del rf, best_rf

# ---------- Isolation Forest (unsupervised): choose on validation AUC ----------
best_if = None
legit = y_train == 0
for fit_on, Xfit in (("all", X_train), ("legit", X_train[legit])):
    for ms in (256, 4096):
        t = time.time()
        iso = IsolationForest(n_estimators=300, max_samples=ms, n_jobs=-1, random_state=42)
        iso.fit(Xfit)
        s_val = -iso.score_samples(X_val)
        auc = roc_auc_score(y_val, s_val)
        print(f"IF fit_on={fit_on} max_samples={ms}: val AUC {auc:.4f} ({time.time()-t:.0f}s)", flush=True)
        if best_if is None or auc > best_if[0]:
            best_if = (auc, fit_on, ms, iso, time.time() - t)
_, fit_on, ms, iso, if_time = best_if
r = evaluate(f"IsolationForest (fit on {fit_on}, max_samples={ms})",
             -iso.score_samples(X_val), -iso.score_samples(X_test), prob=False)
r["train_time_s"] = if_time
results.append(r)

json.dump(results, open("rf_if_results.json", "w"), indent=1, default=float)
print("\nDONE in", f"{time.time()-t0:.0f}s")
