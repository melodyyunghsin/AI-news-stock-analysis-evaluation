"""Backfill the top-level `relevance` and `relevance_reasoning` columns into
existing prediction parquets, by extracting them from any horizon's pred_{h}
JSON.

Idempotent — rows whose top-level relevance is already set are left alone.
Rows that haven't been predicted yet (no pred_{h} JSON anywhere) get None.

⚠️ Pause qwen_predict.py before running — both write to the same files.
"""

import os
import json

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


PRED_DIR = "data/prediction_batches_qwen"
HORIZONS = ["1d", "3d", "5d", "10d", "21d"]


def extract_from_row(row):
    """Return (relevance, reasoning) from any horizon's pred_{h} JSON, or
    (None, None) if no horizon has a parseable prediction with relevance.
    """
    for h in HORIZONS:
        pred_json = row.get(f"pred_{h}")
        if not isinstance(pred_json, str) or not pred_json.strip():
            continue
        try:
            pred = json.loads(pred_json)
        except Exception:
            continue
        rel = pred.get("relevance")
        if rel is None:
            continue
        try:
            return float(rel), str(pred.get("relevance_reasoning", ""))
        except (TypeError, ValueError):
            continue
    return None, None


def backfill_file(path):
    df = pd.read_parquet(path)

    if "relevance" not in df.columns:
        df["relevance"] = None
    if "relevance_reasoning" not in df.columns:
        df["relevance_reasoning"] = None

    n_filled = 0
    n_already = 0
    n_no_pred = 0

    for idx, row in df.iterrows():
        if pd.notna(row.get("relevance")):
            n_already += 1
            continue
        rel, reason = extract_from_row(row)
        if rel is None:
            n_no_pred += 1
            continue
        df.at[idx, "relevance"] = rel
        df.at[idx, "relevance_reasoning"] = reason
        n_filled += 1

    name = os.path.basename(path)
    if n_filled > 0:
        pq.write_table(pa.Table.from_pandas(df), path)
        print(f"  💾 {name}: filled {n_filled}, already had {n_already}, no-pred {n_no_pred}")
    else:
        print(f"  -  {name}: nothing to fill (already {n_already}, no-pred {n_no_pred})")


def main():
    if not os.path.isdir(PRED_DIR):
        raise RuntimeError(f"PRED_DIR not found: {PRED_DIR}")

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".parquet"))
    if not files:
        print(f"No parquets in {PRED_DIR}")
        return

    print(f"📋 Backfilling relevance column into {len(files)} parquet(s)")
    for f in files:
        backfill_file(os.path.join(PRED_DIR, f))
    print("✅ Done")


if __name__ == "__main__":
    main()
