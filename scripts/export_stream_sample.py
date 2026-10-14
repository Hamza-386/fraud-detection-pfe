"""
Exports the first transactions of the test period, raw and in
chronological order, for the dashboard's Live tab:
data/stream_sample.csv.gz (isFraud is kept only to evaluate the stream;
it is never sent to the API with the transaction).

These are exactly the transactions that follow the state saved in
models/online_state.pkl, so streaming them reproduces the notebook's
scores.

Usage: python scripts/export_stream_sample.py [--n 3000]
"""
import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(ROOT, ".."))
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()

tr = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
idt = pd.read_csv(os.path.join(args.data_dir, "train_identity.csv"))
df = tr.merge(idt, on="TransactionID", how="left")
df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
n = len(df)
start = int(n * 0.70) + int(n * 0.10)
out = os.path.join(ROOT, "data", "stream_sample.csv.gz")
df.iloc[start:start + args.n].to_csv(out, index=False, compression="gzip")
print(f"{args.n} test transactions (rows {start}..{start + args.n - 1}) -> {out} "
      f"({os.path.getsize(out) / 1e6:.1f} MB)")
