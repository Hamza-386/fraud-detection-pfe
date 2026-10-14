"""
Online feature engine: computes, for ONE raw transaction, the 512
features the production models expect, exactly as the notebook does
offline (CELLS 7B, 8, 8B, 8C, 9, 10 and the full-data target encoding
of CELL 13B).

The offline pipeline computes "past-only" features with groupby /
cumcount / cumsum over the chronologically sorted data. Online, the
same quantities are kept in a small per-entity state (per card1, uid,
email domain, ...) that is read to compute the features of a new
transaction and then updated with it.

Two kinds of state:
  * behaviour state (counts, amounts, last transaction, ...): updated
    with every scored transaction, since it uses no label;
  * label state (target encoding): sums / counts of known frauds per
    key. It starts from the training statistics saved by the notebook
    and only changes when a label arrives through `add_label`
    (in reality, days later, when a chargeback is confirmed).

With no label feedback, the target encoding is frozen at the end of
the training period, which is exactly how the notebook encoded the
validation and test sets: replaying the data through this engine
reproduces the offline features (see scripts/build_online_state.py).
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants copied from the notebook
# ---------------------------------------------------------------------------
D_NORM_COLS = ["D1", "D2", "D3", "D4", "D10", "D15"]          # CELL 7B
FREQ_COLS = ["card2", "addr1", "P_emaildomain", "R_emaildomain"]  # CELL 8
TOP_V = ["V257", "V246", "V244", "V242", "V201",
         "V200", "V189", "V188", "V258", "V45"]                # CELL 8
TOP_V_WEIGHTS = [0.383, 0.367, 0.364, 0.361, 0.328,
                 0.319, 0.308, 0.304, 0.297, 0.282]
C_HIGH = ["C1", "C2", "C6", "C11"]                             # CELL 8C
D_GROUP = ["D1", "D4", "D10", "D15"]                           # CELL 8C
RISK_COLS = ["is_risky_hour", "is_risky_product", "is_mobile",
             "is_discover", "is_risky_email", "M4_is_M2",
             "C_and_mobile", "S_and_mobile"]                   # CELL 8C
ROLL_WINDOW = 5                                                # CELL 8B
MISSING_CAT = "__MISSING__"                                    # CELL 9

NAN = float("nan")


def _isnan(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v: Any) -> float:
    """Raw value -> float (NaN when missing)."""
    if _isnan(v):
        return NAN
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def _uid_part(v: Any, is_int: bool) -> str:
    """Same string as pandas `.fillna(-999).astype(str)` (CELL 7B).

    card1 has no missing value, so pandas reads it as int64 ("13926");
    the other UID columns are float64 ("361.0", "-999.0")."""
    if _isnan(v):
        return "-999" if is_int else "-999.0"
    return str(int(v)) if is_int else str(float(v))


def _ratio_std(n: int, s: float, sq: float) -> float:
    """Sample std from count / sum / sum of squares, as CELL 8 does
    (NaN for fewer than two values)."""
    if n < 2:
        return NAN
    mean = s / n
    var = (sq - n * mean * mean) / (n - 1)
    return math.sqrt(max(var, 0.0))


class OnlineFeatureEngine:
    """Stateful feature computation for one transaction at a time."""

    def __init__(self, preprocessing: Dict[str, Any], feature_names: List[str]):
        self.feature_names = list(feature_names)
        self.label_mappings: Dict[str, Dict[str, int]] = preprocessing["label_mappings"]
        self.train_medians: Dict[str, float] = preprocessing["train_medians"]
        self.nan_flag_cols: List[str] = preprocessing["nan_flag_cols"]
        self.c13_p75: float = float(preprocessing["c13_p75"])
        self.te_cols: List[str] = preprocessing["te_cols"]
        self.m: float = float(preprocessing["te_smoothing"])
        self.cat_cols = [c for c in self.label_mappings if c != "uid"]

        # Label state: target-encoding sums / counts, seeded with the
        # training statistics. Keys are the processed values the notebook
        # used (label codes / imputed values); the uid is keyed by its
        # string so that customers first seen after training can also
        # accumulate a history from label feedback.
        encoders = preprocessing["te_encoders"]
        self.global_mean = float(encoders["uid"][2])
        self.te_stats: Dict[str, Dict[Any, List[float]]] = {}
        for col in self.te_cols:
            sums, counts, _ = encoders[col]
            self.te_stats[col] = {
                k: [float(s), float(c)] for k, s, c in
                zip(sums.index.tolist(), sums.to_numpy(), counts.to_numpy())
            }
        uid_code_to_str = {code: s for s, code in self.label_mappings["uid"].items()}
        sums, counts, _ = encoders["uid"]
        self.te_stats["uid"] = {
            uid_code_to_str[k]: [float(s), float(c)] for k, s, c in
            zip(sums.index.tolist(), sums.to_numpy(), counts.to_numpy())
            if k in uid_code_to_str
        }

        # Behaviour state (no labels).
        self.card1: Dict[Any, Dict[str, Any]] = {}
        self.uid: Dict[str, List[float]] = {}        # [count, sum, sumsq]
        self.email: Dict[str, List[float]] = {}      # [count, sum]
        self.freq: Dict[str, Dict[Any, int]] = {c: {} for c in ("card2", "addr1", "R_emaildomain")}
        self.n_seen = 0

        # TransactionID -> target-encoding keys, so that a label arriving
        # later can be attributed to the right customer / card / ...
        self.pending: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
        self.max_pending = 200_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, raw: Dict[str, Any], commit: bool = True
                ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Features of one raw transaction (merged transaction + identity
        columns, as in train_transaction.csv / train_identity.csv).

        Returns (features, info). With commit=True the behaviour state is
        updated with this transaction afterwards (normal online use)."""
        f, keys, upd = self._compute(raw)
        info = {"uid": keys["uid"], "is_new_customer": keys["uid"] not in self.te_stats["uid"],
                "uid_history": int(self.uid.get(keys["uid"], [0])[0])}
        if commit:
            self._commit(upd)
            tid = raw.get("TransactionID")
            if tid is not None:
                self.pending[tid] = keys
                if len(self.pending) > self.max_pending:
                    self.pending.popitem(last=False)
        return f, info

    def observe(self, raw: Dict[str, Any]) -> None:
        """Updates the behaviour state with a past transaction without
        computing its features (warm-up on the history)."""
        dt = int(_num(raw["TransactionDT"]))
        email = raw.get("P_emaildomain")
        self._commit({
            "card1": int(_num(raw["card1"])), "amt": _num(raw["TransactionAmt"]),
            "dt": dt, "is_night": int(0 <= dt % 86400 // 3600 <= 6),
            "addr": -999.0 if _isnan(raw.get("addr1")) else _num(raw.get("addr1")),
            "email": "__MISSING_EMAIL__" if _isnan(email) else email,
            "uid": self._uid(raw, dt), "raw": raw})

    def vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features[n] for n in self.feature_names], dtype=np.float64)

    def add_label(self, transaction_id: Any, is_fraud: int) -> bool:
        """Label feedback (e.g. a confirmed chargeback): updates the
        target-encoding statistics of the transaction's keys."""
        keys = self.pending.pop(transaction_id, None)
        if keys is None:
            return False
        y = 1.0 if is_fraud else 0.0
        for col in self.te_cols + ["uid"]:
            st = self.te_stats[col].setdefault(keys[col], [0.0, 0.0])
            st[0] += y
            st[1] += 1.0
        return True

    def state_summary(self) -> Dict[str, int]:
        return {
            "transactions_seen": self.n_seen,
            "cards": len(self.card1),
            "customers": len(self.uid),
            "customers_with_label_history": len(self.te_stats["uid"]),
            "pending_labels": len(self.pending),
        }

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------
    def _te(self, col: str, key: Any) -> float:
        st = self.te_stats[col].get(key)
        if st is None:
            return self.global_mean
        return (st[0] + self.m * self.global_mean) / (st[1] + self.m)

    @staticmethod
    def _uid(raw: Dict[str, Any], dt: int) -> str:
        """Pseudo-customer key of CELL 7B: card fields, address and the
        normalised D1 (day the card was first used)."""
        d1n = _num(raw.get("D1")) - dt // 86400
        d1n_str = "-999.0" if math.isnan(d1n) else str(float(np.round(d1n, 0)))
        return "_".join([_uid_part(raw.get("card1"), True)]
                        + [_uid_part(raw.get(c), False)
                           for c in ("card2", "card3", "card5", "addr1", "addr2")]
                        + [d1n_str])

    def _compute(self, raw: Dict[str, Any]):
        f: Dict[str, float] = {}
        dt = int(_num(raw["TransactionDT"]))
        amt = _num(raw["TransactionAmt"])
        card1 = int(_num(raw["card1"]))

        # ---- raw numeric columns (everything that is not categorical)
        for name in self.feature_names:
            if name in raw and name not in self.label_mappings:
                f[name] = _num(raw[name])

        # ---- CELL 7B: D normalisation and UID
        dt_day = dt // 86400
        f["DT_day"] = dt_day
        for c in D_NORM_COLS:
            f[f"{c}_norm"] = _num(raw.get(c)) - dt_day
        uid = self._uid(raw, dt)

        # ---- CELL 8: time / amount
        hour = dt % 86400 // 3600
        f["hour"] = hour
        f["day"] = dt // 86400 % 7
        f["week"] = dt // 86400 // 7
        is_night = int(0 <= hour <= 6)
        f["is_night"] = is_night
        f["is_morning"] = int(4 <= hour <= 11)
        f["is_weekend"] = int(f["day"] in (5, 6))
        f["Amt_log"] = math.log1p(amt)
        f["Amt_sqrt"] = math.sqrt(amt)
        f["Amt_is_round"] = int(amt % 1 == 0)
        f["Amt_cents"] = amt % 1

        # ---- CELL 8: card1 history
        c1 = self.card1.get(card1)
        n1 = c1["n"] if c1 else 0
        f["card1_freq"] = n1
        f["card1_count"] = n1
        if n1 == 0:
            prev_mean, prev_std = amt, 0.0
        else:
            prev_mean = c1["sum"] / n1
            std = _ratio_std(n1, c1["sum"], c1["sumsq"])
            prev_std = 0.0 if math.isnan(std) else std
        f["Amt_vs_card1_mean"] = amt / (prev_mean + 1e-6)
        f["Amt_vs_card1_std"] = amt / (prev_std + 1)

        # ---- CELL 8: frequency of other keys (NaN key -> NaN, as cumcount)
        for col in FREQ_COLS:
            v = raw.get(col)
            if _isnan(v):
                f[f"{col}_freq"] = NAN
            elif col == "P_emaildomain":
                f[f"{col}_freq"] = self.email.get(v, [0, 0.0])[0]
            else:
                key = _num(v) if col != "R_emaildomain" else v
                f[f"{col}_freq"] = self.freq[col].get(key, 0)
        f["card2_count"] = f["card2_freq"]
        f["addr1_count"] = f["addr1_freq"]
        email = raw.get("P_emaildomain")
        if _isnan(email):
            f["email_count"] = NAN
            f["email_amt_mean"] = amt
        else:
            en, es = self.email.get(email, [0, 0.0])
            f["email_count"] = en
            f["email_amt_mean"] = es / en if en else amt

        # ---- CELL 8: top V aggregates (pandas skipna semantics)
        vv = [_num(raw.get(v)) for v in TOP_V]
        present = [x for x in vv if not math.isnan(x)]
        f["V_top_sum"] = float(sum(present))
        f["V_top_mean"] = float(np.mean(present)) if present else NAN
        f["V_top_std"] = float(np.std(present, ddof=1)) if len(present) > 1 else NAN
        f["V_top_max"] = max(present) if present else NAN
        f["V_top_min"] = min(present) if present else NAN
        f["V_weighted_score"] = float(sum(x * w for x, w in zip(vv, TOP_V_WEIGHTS)))

        # ---- CELL 8: binary risk flags
        product, device = raw.get("ProductCD"), raw.get("DeviceType")
        card4 = raw.get("card4")
        f["is_risky_hour"] = int(4 <= hour <= 11)
        f["is_risky_product"] = int(product in ("C", "S"))
        f["is_mobile"] = int(device == "mobile")
        f["is_discover"] = int(card4 == "discover")
        f["is_risky_email"] = int(email in ("outlook.com", "hotmail.com", "gmail.com"))

        # ---- CELL 8: UID history
        u = self.uid.get(uid)
        un = int(u[0]) if u else 0
        f["uid_freq"] = un
        f["uid_count"] = un
        uid_mean = u[1] / un if un else amt
        f["uid_amt_mean"] = uid_mean
        if un >= 2:
            var = (u[2] - un * uid_mean * uid_mean) / (un - 1)
            f["uid_amt_std"] = math.sqrt(max(var, 0.0))
        else:
            f["uid_amt_std"] = 0.0
        f["Amt_vs_uid_mean"] = amt / (uid_mean + 1e-6)

        # ---- CELL 8B: lags, rolling window, interactions, diversity
        if n1:
            f["card1_prev_amt"] = c1["last_amt"]
            f["card1_time_diff"] = dt - c1["last_dt"]
            window = list(c1["window"])
            f["card1_roll_mean"] = float(np.mean(window))
            f["card1_roll_std"] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            f["card1_night_ratio"] = c1["nights"] / n1
            f["card1_unique_addr"] = len(c1["addrs"])
            f["card1_unique_email"] = len(c1["emails"])
        else:
            f["card1_prev_amt"] = amt
            f["card1_time_diff"] = 0
            f["card1_roll_mean"] = amt
            f["card1_roll_std"] = 0.0
            f["card1_night_ratio"] = 0.0
            f["card1_unique_addr"] = 0
            f["card1_unique_email"] = 0
        f["amt_vs_prev"] = amt / (f["card1_prev_amt"] + 1)
        f["amt_zscore_roll"] = (amt - f["card1_roll_mean"]) / (f["card1_roll_std"] + 1)
        f["hour_x_product"] = hour * f["is_risky_product"]
        f["amt_x_mobile"] = amt * f["is_mobile"]
        f["card1_freq_x_amt"] = n1 * amt
        f["email_risk_x_amt"] = f["is_risky_email"] * amt

        # ---- CELL 8C: combinations and groups
        m4 = raw.get("M4")
        f["M4_is_M2"] = int(m4 == "M2")
        f["M4_is_M0"] = int(m4 == "M0")
        f["C_and_mobile"] = int(product == "C" and device == "mobile")
        f["S_and_mobile"] = int(product == "S" and device == "mobile")
        f["R_and_mobile"] = int(product == "R" and device == "mobile")
        f["discover_and_S"] = int(card4 == "discover" and product == "S")
        f["amex_and_C"] = int(card4 == "american express" and product == "C")
        f["visa_and_C"] = int(card4 == "visa" and product == "C")
        ch = [x for x in (_num(raw.get(c)) for c in C_HIGH) if not math.isnan(x)]
        f["C_high_sum"] = float(sum(ch))
        f["C_high_mean"] = float(np.mean(ch)) if ch else NAN
        f["C13_high"] = int(_num(raw.get("C13")) > self.c13_p75)
        dg = [x for x in (_num(raw.get(c)) for c in D_GROUP) if not math.isnan(x)]
        f["D_sum"] = float(sum(dg))
        f["D_mean"] = float(np.mean(dg)) if dg else NAN
        f["total_risk_score"] = sum(f[c] for c in RISK_COLS)

        # ---- CELL 9: NaN flags on the raw values
        for col in self.nan_flag_cols:
            f[f"{col}_nan_flag"] = int(_isnan(raw.get(col)))

        # ---- CELL 9: label encoding learned on train (unknown -> -1)
        for col in self.cat_cols:
            v = raw.get(col)
            s = MISSING_CAT if _isnan(v) else str(v)
            f[col] = self.label_mappings[col].get(s, -1)

        # ---- CELL 10: train-median imputation
        for name in self.feature_names:
            v = f.get(name)
            if v is not None and isinstance(v, float) and math.isnan(v):
                f[name] = self.train_medians.get(name, 0.0)

        # ---- CELL 13B: target encoding (replaces the raw key columns)
        keys = {col: f[col] for col in self.te_cols}
        keys["uid"] = uid
        for col in self.te_cols:
            f[col] = self._te(col, keys[col])
        f["uid_target_enc"] = self._te("uid", uid)

        missing = [n for n in self.feature_names if n not in f]
        if missing:
            raise KeyError(f"features not computed: {missing[:10]}")

        upd = {"card1": card1, "amt": amt, "dt": dt, "is_night": is_night,
               "addr": -999.0 if _isnan(raw.get("addr1")) else _num(raw.get("addr1")),
               "email": "__MISSING_EMAIL__" if _isnan(email) else email,
               "uid": uid, "raw": raw}
        return f, keys, upd

    # ------------------------------------------------------------------
    # State update (after scoring)
    # ------------------------------------------------------------------
    def _commit(self, upd: Dict[str, Any]) -> None:
        amt, raw = upd["amt"], upd["raw"]
        c1 = self.card1.get(upd["card1"])
        if c1 is None:
            c1 = {"n": 0, "sum": 0.0, "sumsq": 0.0, "last_amt": None, "last_dt": None,
                  "window": deque(maxlen=ROLL_WINDOW), "nights": 0,
                  "addrs": set(), "emails": set()}
            self.card1[upd["card1"]] = c1
        c1["n"] += 1
        c1["sum"] += amt
        c1["sumsq"] += amt * amt
        c1["last_amt"] = amt
        c1["last_dt"] = upd["dt"]
        c1["window"].append(amt)
        c1["nights"] += upd["is_night"]
        c1["addrs"].add(upd["addr"])
        c1["emails"].add(upd["email"])

        u = self.uid.setdefault(upd["uid"], [0, 0.0, 0.0])
        u[0] += 1
        u[1] += amt
        u[2] += amt * amt

        email = raw.get("P_emaildomain")
        if not _isnan(email):
            e = self.email.setdefault(email, [0, 0.0])
            e[0] += 1
            e[1] += amt
        for col in ("card2", "addr1", "R_emaildomain"):
            v = raw.get(col)
            if not _isnan(v):
                key = _num(v) if col != "R_emaildomain" else v
                self.freq[col][key] = self.freq[col].get(key, 0) + 1
        self.n_seen += 1


def iter_records(df, chunk: int = 20_000) -> Iterable[Dict[str, Any]]:
    """Rows of a DataFrame as dicts with NaN for missing values."""
    for start in range(0, len(df), chunk):
        for rec in df.iloc[start:start + chunk].to_dict("records"):
            yield rec
