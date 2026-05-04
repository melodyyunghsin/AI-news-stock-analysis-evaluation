"""Backfill the majority-direction columns into existing prediction parquets.

After adding `actual_majority_direction_{h}`, `actual_up_days_{h}`, and
`actual_down_days_{h}` to compute_returns_50_with_horizon.py, parquets that
were already in PRED_DIR (data/prediction_batches_qwen) won't have them.
This script computes the columns for every row in every parquet under
PRED_DIR using the same logic as the upstream return-batch compute.

Idempotent — files that already have the columns are left alone. Safe to run
on partially-predicted parquets; the new columns are derived from price data,
not from predictions.

⚠️ Pause qwen_predict.py before running this — both write to the same files
   under PRED_DIR and a concurrent run can corrupt a checkpoint.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from compute_returns_50_with_horizon import (
    HORIZONS,
    get_majority_direction_from_csv,
)

PRED_DIR = "data/prediction_batches_qwen"


def backfill_file(path):
    df = pd.read_parquet(path)

    if "Stock_symbol" not in df.columns or "Date" not in df.columns:
        print(f"  ⚠️ {os.path.basename(path)} missing Stock_symbol/Date — skipping")
        return

    changed = False
    for label, window in HORIZONS.items():
        maj_col = f"actual_majority_direction_{label}"
        up_col = f"actual_up_days_{label}"
        down_col = f"actual_down_days_{label}"

        if maj_col in df.columns and up_col in df.columns and down_col in df.columns:
            continue  # already present

        majs, ups, downs = [], [], []
        for _, row in df.iterrows():
            ticker = str(row["Stock_symbol"]).strip()
            d = row["Date"]
            if isinstance(d, (int, float)):
                date_str = pd.to_datetime(d, unit="ms").strftime("%Y-%m-%d")
            else:
                date_str = str(d)[:10]
            maj, u, dn = get_majority_direction_from_csv(ticker, date_str, window)
            majs.append(maj)
            ups.append(u)
            downs.append(dn)

        df[maj_col] = majs
        df[up_col] = ups
        df[down_col] = downs
        changed = True
        print(f"  + {label}")

    if changed:
        pq.write_table(pa.Table.from_pandas(df), path)
        print(f"  💾 saved {os.path.basename(path)}")
    else:
        print(f"  - {os.path.basename(path)}: already up to date")


def main():
    if not os.path.isdir(PRED_DIR):
        raise RuntimeError(f"Prediction dir not found: {PRED_DIR}")

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".parquet"))
    if not files:
        print(f"No parquets found in {PRED_DIR}")
        return

    print(f"📈 Backfilling majority-direction columns into {len(files)} parquet(s)")
    for f in files:
        backfill_file(os.path.join(PRED_DIR, f))
    print("✅ Done")


if __name__ == "__main__":
    main()
