"""
Reclassify actual_direction and actual_strength in all return batch parquet files.

New thresholds:
  |return| <= 1%          -> UP/DOWN, weak
  1%  < |return| <= 5%   -> UP/DOWN, moderate
  |return| > 5%           -> UP/DOWN, strong
  return is None/NaN      -> NO_DATA, none
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

RETURN_DIR = "data/return_batches"
HORIZONS   = ["1d", "3d", "5d", "10d", "21d"]


def classify(pct):
    if pct is None or pd.isna(pct):
        return "NO_DATA", "none"
    ap = abs(pct)
    direction = "UP" if pct > 0 else "DOWN"
    if ap <= 1.0:
        return direction, "weak"
    if ap <= 5.0:
        return direction, "moderate"
    return direction, "strong"


def reclassify_file(path):
    df = pd.read_parquet(path)
    changed = False

    for h in HORIZONS:
        ret_col = f"return_{h}"
        dir_col = f"actual_direction_{h}"
        str_col = f"actual_strength_{h}"

        if ret_col not in df.columns:
            continue

        dirs, strs = zip(*df[ret_col].map(classify)) if len(df) > 0 else ([], [])
        df[dir_col] = list(dirs)
        df[str_col] = list(strs)
        changed = True

    if changed:
        pq.write_table(pa.Table.from_pandas(df), path)

    return changed, len(df)


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No parquet files found in", RETURN_DIR)
        return

    total_rows = 0
    for i, fname in enumerate(files, 1):
        path = os.path.join(RETURN_DIR, fname)
        changed, n = reclassify_file(path)
        total_rows += n
        status = "updated" if changed else "skipped (no return columns)"
        print(f"[{i}/{len(files)}] {status}: {fname} ({n} rows)")

    print(f"\nDone. {len(files)} files, {total_rows} total rows.")


if __name__ == "__main__":
    main()
