"""Smaller-subset evaluation on gemini-2.5-pro.

Imports the full pipeline from gemini_predict.py and only overrides config:
  - MODEL: gemini-2.5-pro instead of gemini-2.5-flash-lite
  - Output dirs: data/pro_test_*  (so flash-lite results stay untouched)
  - TICKER_PRIORITY: small subset for cost control
  - MAX_ARTICLES_PER_TICKER: tighter cap (40 → 10/quadrant)

Defaults to AAPL + TSLA: AAPL is the flash-lite pipeline's strongest ticker,
TSLA its weakest. Comparing both shows whether pro lifts both ends or just one.
Edit TICKER_PRIORITY below to change the subset.

Run:
    export GEMINI_API_KEYS=<your-keys>
    python gemini_predict_pro.py
"""
import os
import gemini_predict as gp

# ============================================================
# CONFIG OVERRIDES
# ============================================================

gp.MODEL = "models/gemini-2.5-pro"
gp.PRED_DIR = "data/pro_test_predictions_gemini_pro"
gp.EVAL_DIR = "data/pro_test_evaluation_results_gemini_pro"
gp.TICKER_PRIORITY = ["AAPL", "TSLA", "AMZN"]  # AMZN included since the leftover-bug already ran it
gp._TICKER_RANK = {t: i for i, t in enumerate(gp.TICKER_PRIORITY)}
gp.MAX_ARTICLES_PER_TICKER = 40  # 10 per quadrant; vs 100 in the main pipeline

# Make the new dirs (gemini_predict already made the originals at import)
os.makedirs(gp.PRED_DIR, exist_ok=True)
os.makedirs(gp.EVAL_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("GEMINI 2.5 PRO — SUBSET TEST RUN")
    print("=" * 70)
    print(f"  Model:               {gp.MODEL}")
    print(f"  Tickers:             {gp.TICKER_PRIORITY}")
    print(f"  Max articles/ticker: {gp.MAX_ARTICLES_PER_TICKER} (10 per quadrant)")
    print(f"  PRED_DIR:            {gp.PRED_DIR}")
    print(f"  EVAL_DIR:            {gp.EVAL_DIR}")
    print(f"  K_SAMPLES:           {gp.K_SAMPLES}")
    print(f"  USE_BATCH_API:       {gp.USE_BATCH_API}")
    print(f"  4-way tone balance:  {gp.FOUR_WAY_TONE_BALANCED_SELECTION}")
    print("=" * 70)
    print()

    # Replicate gp.main() but DROP the "leftover" tickers — the upstream
    # main() processes any ticker found in the source dir even when it isn't
    # in TICKER_PRIORITY (the priority list was originally an *ordering* hint,
    # not a filter). For the subset test we strictly want only the listed
    # tickers.
    files = sorted(f for f in os.listdir(gp.RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No return batch parquet files found.")
        return

    pipeline_label = "enhanced" if gp.USE_ENHANCED_PIPELINE else "legacy"
    transport_label = " + batch API" if (gp.USE_ENHANCED_PIPELINE and gp.USE_BATCH_API) else ""
    print(f"Pipeline: {pipeline_label}{transport_label}")

    ticker_files = gp.index_files_by_ticker(files)
    ticker_order = [t for t in gp.TICKER_PRIORITY if t in ticker_files]
    missing = [t for t in gp.TICKER_PRIORITY if t not in ticker_files]
    if missing:
        print(f"⚠️  Tickers requested but not found in {gp.RETURN_DIR}: {missing}")

    print(f"\nTicker processing order ({len(ticker_order)} total):")
    for i, t in enumerate(ticker_order, 1):
        print(f"  {i:>3}. {t}  — {len(ticker_files[t])} batches")

    for i, ticker in enumerate(ticker_order, 1):
        print(f"\n🎯 [{i}/{len(ticker_order)}] Ticker: {ticker}")
        if gp.USE_ENHANCED_PIPELINE:
            gp.process_ticker_enhanced(ticker, ticker_files[ticker])
        else:
            for f in ticker_files[ticker]:
                return_path = os.path.join(gp.RETURN_DIR, f)
                gp.predict_file_legacy(return_path, target_ticker=ticker)

    # Eval CSVs only for the requested tickers' source files
    print("\n📄 Writing evaluation CSVs...")
    requested_files = sorted({f for t in ticker_order for f in ticker_files[t]})
    for f in requested_files:
        gp.write_eval_csv(os.path.join(gp.RETURN_DIR, f))


if __name__ == "__main__":
    main()
