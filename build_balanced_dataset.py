"""Build a UP/DOWN-balanced focused dataset.

Pipeline:
  1. Collect every article from RETURN_DIR matching TARGET_TICKERS that has
     1d ground truth.
  2. Score relevance via Qwen for every candidate (cache-backed — articles
     scored by previous runs are free).
  3. Filter to relevance >= RELEVANCE_THRESHOLD (default 0.8).
  4. Per ticker, balance UP/DOWN: keep all of the smaller class, then pick
     the top-N rows of the larger class by relevance, where N == |smaller|.
  5. Save one parquet per ticker to OUT_DIR.

Reuses helpers from build_focused_dataset.py — same 8 tickers, same cache,
same source data.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

import build_focused_dataset as bfd


# ============================================================
# CONFIG
# ============================================================

# Same 8 tickers as bfd.TARGET_TICKERS — listed here for clarity.
TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "GOOG", "TSM"]

RELEVANCE_THRESHOLD = 0.8

# Which horizon's actual_direction column drives the UP/DOWN balance.
DIRECTION_COL = "actual_direction_1d"

OUT_DIR = "data/balanced_focused_dataset"

# Cap candidates per ticker before scoring. None = score every candidate
# (slow when the cache is mostly empty — see bfd notes). Set to e.g. 1000
# to bound LLM cost; cached articles are kept first, so prior runs are reused.
MAX_CANDIDATES_PER_TICKER = None


# Make bfd's helpers write to our output dir / use our ticker list.
bfd.TARGET_TICKERS = TARGET_TICKERS
bfd.FOCUSED_DIR = OUT_DIR


# ============================================================
# Step 5: per-ticker UP/DOWN balance
# ============================================================

def select_balanced(df, direction_col):
    """For each ticker, return equal counts of UP and DOWN rows.

    Keeps all rows of the smaller class. From the larger class, picks the
    top-N rows by relevance, where N == |smaller class|.
    """
    pieces = []
    for ticker in TARGET_TICKERS:
        sub = df[df["Stock_symbol"].astype(str).str.strip() == ticker].copy()
        if len(sub) == 0:
            print(f"  {ticker}: 0 candidates after relevance filter — skipping")
            continue

        sub["_dir"] = sub[direction_col].astype(str).str.upper().str.strip()
        ups = sub[sub["_dir"] == "UP"]
        downs = sub[sub["_dir"] == "DOWN"]
        n_up, n_down = len(ups), len(downs)
        keep = min(n_up, n_down)

        if keep == 0:
            print(
                f"  {ticker}: cannot balance (UP={n_up}, DOWN={n_down}) — "
                f"one class is empty, skipping"
            )
            continue

        ups_top = ups.sort_values(
            "_relevance", ascending=False, kind="mergesort"
        ).head(keep)
        downs_top = downs.sort_values(
            "_relevance", ascending=False, kind="mergesort"
        ).head(keep)

        balanced = pd.concat([ups_top, downs_top], ignore_index=True)
        balanced = balanced.drop(columns=["_dir"])
        pieces.append(balanced)

        print(
            f"  {ticker}: pre-balance UP={n_up} DOWN={n_down} "
            f"-> balanced {keep}+{keep}={2 * keep}"
        )

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ============================================================
# Main
# ============================================================

def main():
    print(f"📋 Building BALANCED focused dataset for {len(TARGET_TICKERS)} tickers")
    print(f"   Relevance threshold: >= {RELEVANCE_THRESHOLD}")
    print(f"   Balancing on: {DIRECTION_COL}")
    print(f"   Output dir: {OUT_DIR}\n")

    print("Step 1/5: Collect candidates with 1d ground truth")
    candidates = bfd.collect_candidates()
    print(f"  Total candidates: {len(candidates)}")
    for ticker in TARGET_TICKERS:
        n = (candidates["Stock_symbol"].astype(str).str.strip() == ticker).sum()
        print(f"    {ticker}: {n}")

    print("\nStep 2/5: Load relevance cache")
    cache = bfd.load_relevance_cache()
    print(f"  Cached scores available for {len(cache)} articles")

    if MAX_CANDIDATES_PER_TICKER:
        print(
            f"\nStep 3/5: Cap to {MAX_CANDIDATES_PER_TICKER} per ticker "
            f"(cached first, recent uncached after)"
        )
        candidates = bfd.cap_per_ticker(candidates, cache, MAX_CANDIDATES_PER_TICKER)
        print(f"  After cap: {len(candidates)}")
    else:
        print("\nStep 3/5: No cap — scoring every candidate")

    print("\nStep 4/5: Score relevance on uncached articles")
    scored = bfd.score_all(candidates, cache)

    print(f"\nStep 5a/5: Filter to relevance >= {RELEVANCE_THRESHOLD}")
    above = scored[scored["_relevance"] >= RELEVANCE_THRESHOLD].copy()
    print(f"  Pool after filter: {len(above)} / {len(scored)}")
    for ticker in TARGET_TICKERS:
        n = (above["Stock_symbol"].astype(str).str.strip() == ticker).sum()
        print(f"    {ticker}: {n}")

    print(f"\nStep 5b/5: Balance UP/DOWN per ticker (column: {DIRECTION_COL})")
    if DIRECTION_COL not in above.columns:
        raise RuntimeError(
            f"{DIRECTION_COL} not found in candidate columns — "
            f"available actual_direction_* columns: "
            f"{[c for c in above.columns if c.startswith('actual_direction_')]}"
        )
    balanced = select_balanced(above, DIRECTION_COL)
    print(f"  Total selected: {len(balanced)}")

    print(f"\nWriting balanced dataset to {OUT_DIR}/")
    bfd.save_per_ticker(balanced, OUT_DIR)

    print("\n✅ Balanced dataset built. Next steps:")
    print("   1. In qwen_predict.py:")
    print(f"        RETURN_DIR          = '{OUT_DIR}'")
    print("        PRED_DIR            = 'data/balanced_predictions_qwen_k5'")
    print("        K_SAMPLES           = 5")
    print("        RELEVANCE_THRESHOLD = 0.0   # already filtered upstream")
    print("   2. Run: python3 qwen_predict.py")
    print("   3. In merge_evaluation.py:")
    print("        PRED_DIR = 'data/balanced_predictions_qwen_k5'")
    print("      then run it.")


if __name__ == "__main__":
    main()
