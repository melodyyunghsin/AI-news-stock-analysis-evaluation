"""Quick diagnostic for build_focused_dataset.py — figures out why candidates
or relevance scores aren't showing up where expected.

Just run: python3 diagnose_focused_dataset.py
"""

import os
import json
import glob

import pandas as pd

RETURN_DIR = "data/return_batches_clean"
PRED_DIR = "data/prediction_batches_qwen"
PRICE_DIR = "data/full_history"
TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "GOOG", "TSM"]
HORIZONS = ["1d", "3d", "5d", "10d", "21d"]


def section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ---------- 1. Price CSVs ----------
section("1. Price CSVs (data/full_history/)")
for t in TARGET_TICKERS:
    path = os.path.join(PRICE_DIR, f"{t}.csv")
    if os.path.exists(path):
        size = os.path.getsize(path)
        try:
            df = pd.read_csv(path)
            print(f"  ✅ {t}.csv  ({size:>10} bytes, {len(df):>5} rows)")
        except Exception as e:
            print(f"  ⚠️ {t}.csv exists but unreadable: {e}")
    else:
        print(f"  ❌ {t}.csv MISSING — every {t} article will get actual_direction='NO_DATA'")


# ---------- 2. return_batches_clean schema + per-ticker counts ----------
section("2. data/return_batches_clean/ — candidate pool")
files = sorted(glob.glob(os.path.join(RETURN_DIR, "*.parquet")))
if not files:
    print(f"  ❌ no parquets in {RETURN_DIR}")
else:
    df0 = pd.read_parquet(files[0])
    print(f"  Sample file: {os.path.basename(files[0])}")
    print(f"  Article_id dtype: {df0['Article_id'].dtype if 'Article_id' in df0.columns else 'MISSING'}")
    print(f"  has actual_direction_1d column: {'actual_direction_1d' in df0.columns}")

    total = {t: 0 for t in TARGET_TICKERS}
    with_gt = {t: 0 for t in TARGET_TICKERS}
    no_data = {t: 0 for t in TARGET_TICKERS}
    null_gt = {t: 0 for t in TARGET_TICKERS}

    has_gt_col = "actual_direction_1d" in df0.columns

    for f in files:
        df = pd.read_parquet(f)
        sub = df[df["Stock_symbol"].astype(str).str.strip().isin(TARGET_TICKERS)]
        for t in TARGET_TICKERS:
            st = sub[sub["Stock_symbol"].astype(str).str.strip() == t]
            total[t] += len(st)
            if has_gt_col:
                gt = st["actual_direction_1d"].astype(str).str.upper()
                with_gt[t] += int(gt.isin(("UP", "DOWN")).sum())
                no_data[t] += int((gt == "NO_DATA").sum())
                null_gt[t] += int(st["actual_direction_1d"].isna().sum())

    print(f"\n  {'Ticker':>8}  {'Total':>8}  {'With UP/DOWN':>14}  {'NO_DATA':>10}  {'NaN':>6}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*10}  {'-'*6}")
    for t in TARGET_TICKERS:
        print(f"  {t:>8}  {total[t]:>8}  {with_gt[t]:>14}  {no_data[t]:>10}  {null_gt[t]:>6}")


# ---------- 3. prediction_batches_qwen — relevance availability ----------
section("3. data/prediction_batches_qwen/ — relevance availability for target tickers")
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.parquet")))
if not pred_files:
    print(f"  ❌ no parquets in {PRED_DIR}")
else:
    df0 = pd.read_parquet(pred_files[0])
    has_rel_col = "relevance" in df0.columns
    print(f"  Sample file has 'relevance' column: {has_rel_col}")
    print(f"  Sample Article_id dtype: {df0['Article_id'].dtype if 'Article_id' in df0.columns else 'MISSING'}")

    n_target_rows = 0
    n_col_filled = 0
    n_json_has_relevance = 0
    n_predicted = 0  # has at least one pred_{h}

    for f in pred_files:
        df = pd.read_parquet(f)
        sub = df[df["Stock_symbol"].astype(str).str.strip().isin(TARGET_TICKERS)]
        n_target_rows += len(sub)
        if "relevance" in sub.columns:
            n_col_filled += int(sub["relevance"].notna().sum())
        for _, row in sub.iterrows():
            has_pred = False
            json_has_rel = False
            for h in HORIZONS:
                pred_json = row.get(f"pred_{h}")
                if isinstance(pred_json, str) and pred_json.strip():
                    has_pred = True
                    try:
                        pred = json.loads(pred_json)
                        if pred.get("relevance") is not None:
                            json_has_rel = True
                            break
                    except Exception:
                        pass
            if has_pred:
                n_predicted += 1
            if json_has_rel:
                n_json_has_relevance += 1

    print(f"\n  Target-ticker rows in pred parquets:                {n_target_rows}")
    print(f"  Rows with predictions (any horizon):                {n_predicted}")
    print(f"  Rows with relevance column populated:               {n_col_filled}")
    print(f"  Rows with relevance in pred_{{h}} JSON:               {n_json_has_relevance}")
    if n_predicted > 0 and n_col_filled == 0 and n_json_has_relevance == 0:
        print(f"\n  ⚠️ Predicted rows exist but neither column nor JSON has relevance.")
        print(f"     This is unexpected — predictions should always include relevance.")
    elif n_predicted > 0 and n_col_filled == 0 and n_json_has_relevance > 0:
        print(f"\n  ℹ️ Relevance is in JSON but column is empty — run backfill_relevance_column.py.")


# ---------- 4. Article_id type-match check ----------
section("4. Article_id cross-file consistency")
if files and pred_files:
    rdf = pd.read_parquet(files[0])
    pdf = pd.read_parquet(pred_files[0])
    if "Article_id" in rdf.columns and "Article_id" in pdf.columns:
        r_sample = rdf["Article_id"].head(3).tolist()
        p_sample = pdf["Article_id"].head(3).tolist()
        print(f"  return_batches_clean sample IDs:  {r_sample}  (dtype {rdf['Article_id'].dtype})")
        print(f"  prediction_batches_qwen sample:   {p_sample}  (dtype {pdf['Article_id'].dtype})")
        r_strs = [str(x) for x in r_sample]
        p_strs = [str(x) for x in p_sample]
        if rdf["Article_id"].dtype != pdf["Article_id"].dtype:
            print(f"  ⚠️ dtype mismatch — str() may produce different keys "
                  f"(e.g. '123' vs '123.0').")
        else:
            print(f"  ✅ same dtype — str() should produce matching cache keys.")


# ---------- 5. Summary ----------
section("5. Likely root causes")
print("  (cross-reference with sections 1–4 above)")
print("  • If a ticker shows 0 with-GT in section 2 but has price CSV (section 1):")
print("    → price CSV doesn't span the article dates. Re-check the CSV's date range.")
print("  • If a ticker is missing the price CSV (section 1):")
print("    → every article gets NO_DATA. Add the CSV or remove the ticker.")
print("  • If section 2 says 'has actual_direction_1d column: False':")
print("    → run compute_returns_50_with_horizon.py with RETURN_DIR set to")
print("      'data/return_batches_clean' (it currently points at 'data/return_batches').")
print("  • If section 3 shows column=0 but JSON>0: run backfill_relevance_column.py.")
print("  • If Article_id dtype mismatch in section 4: cache lookups silently miss.")
