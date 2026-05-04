"""Build a focused dataset of top-relevance articles for self-consistency evaluation.

Pipeline:
  1. Pull every article from RETURN_DIR matching TARGET_TICKERS that has 1d ground truth.
  2. Score relevance on each (one factor-extraction LLM call per article).
     - Reuses scores already stored in EXISTING_PRED_DIR.
     - Periodically writes a parquet cache so a crash doesn't lose work.
  3. Select top N per ticker by relevance.
  4. Save selected articles to FOCUSED_DIR as one parquet per ticker.

After this completes:
  - In qwen_predict.py:
        RETURN_DIR        = "data/focused_dataset"
        PRED_DIR          = "data/focused_predictions_qwen_k5"
        K_SAMPLES         = 5
        RELEVANCE_THRESHOLD = 0.0   # don't gate inside qwen_predict — we already
                                     # filtered by relevance during selection
  - Run: python3 qwen_predict.py
  - Then update merge_evaluation.py PRED_DIR and run it.

Time estimates (local Ollama, ~5s per call):
  - Step 2 (relevance scoring):  ~4-12 hours depending on cache hit rate
  - Self-consistency prediction: ~2 days for 1600 articles × 26 calls each
"""

import os
import json

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

import qwen_predict


# ============================================================
# CONFIG
# ============================================================

TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "GOOG", "TSM"]
TOP_N_PER_TICKER = 200

# Cap per-ticker candidates BEFORE relevance scoring. Cached articles are kept
# first; remaining slots filled with the most-recent uncached articles.
# Set to None to score every candidate (slow — ~85h with mostly empty cache).
# 1000 -> ~10-13 hours, picking top 200 from a 1000-article pool per ticker.
#  500 -> ~5-6 hours, picking top 200 from 500 (3:2 selectivity).
MAX_CANDIDATES_PER_TICKER = 1000

RETURN_DIR = "data/return_batches_clean"
EXISTING_PRED_DIR = "data/prediction_batches_qwen"
FOCUSED_DIR = "data/focused_dataset"
RELEVANCE_CACHE_PATH = "data/focused_relevance_cache.parquet"

CHECKPOINT_EVERY = 100  # save cache every N newly-scored articles

os.makedirs(FOCUSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RELEVANCE_CACHE_PATH) or ".", exist_ok=True)


# ============================================================
# Article key — content-based, since Article_id may not exist
# ============================================================

def make_article_key(row):
    """Stable cross-file key built from (ticker, date, article_text[:200]).

    Used in place of Article_id, which isn't present in this dataset's parquets.
    The text snippet makes the key unique even when a ticker has multiple
    articles on the same day.
    """
    ticker = str(row.get("Stock_symbol", "")).strip()
    d = row.get("Date")
    if isinstance(d, (int, float)):
        date_str = pd.to_datetime(d, unit="ms").strftime("%Y-%m-%d")
    else:
        date_str = str(d)[:10]
    text_snippet = str(row.get("Article_text", ""))[:200].strip()
    return (ticker, date_str, text_snippet)


# ============================================================
# Step 1: collect candidate articles
# ============================================================

def collect_candidates():
    """Articles from target tickers that have 1d ground truth."""
    if not os.path.isdir(RETURN_DIR):
        raise RuntimeError(f"RETURN_DIR not found: {RETURN_DIR}")

    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    chunks = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(RETURN_DIR, f))
        except Exception as e:
            print(f"  ⚠️ skip {f}: {e}")
            continue
        sub = df[df["Stock_symbol"].astype(str).str.strip().isin(TARGET_TICKERS)].copy()
        if len(sub) == 0:
            continue
        if "actual_direction_1d" in sub.columns:
            sub = sub.dropna(subset=["actual_direction_1d"])
            sub = sub[
                sub["actual_direction_1d"].astype(str).str.upper().isin(("UP", "DOWN"))
            ]
        sub["_source_file"] = f
        chunks.append(sub)

    if not chunks:
        raise RuntimeError("No candidate articles found in any return-batch parquet.")
    return pd.concat(chunks, ignore_index=True)


# ============================================================
# Step 2: relevance cache + scoring
# ============================================================

def load_relevance_cache():
    """Build {(ticker, date_str, text_snippet): (relevance, reasoning)} from prior runs.

    Sources, in order (later overwrites earlier):
      1. Existing prediction parquets in EXISTING_PRED_DIR — relevance was stored
         on every prediction (in the new top-level column or in pred_{h} JSON).
      2. This script's own checkpoint cache at RELEVANCE_CACHE_PATH.
    """
    cache = {}

    # 1. Existing predictions
    if os.path.isdir(EXISTING_PRED_DIR):
        files = sorted(f for f in os.listdir(EXISTING_PRED_DIR) if f.endswith(".parquet"))
        for f in files:
            try:
                df = pd.read_parquet(os.path.join(EXISTING_PRED_DIR, f))
            except Exception:
                continue
            for _, row in df.iterrows():
                ticker = str(row.get("Stock_symbol", "")).strip()
                if ticker not in TARGET_TICKERS:
                    continue
                key = make_article_key(row)
                if key in cache:
                    continue

                # Prefer the dedicated relevance column (canonical post-refactor)
                rel_col = row.get("relevance")
                if pd.notna(rel_col):
                    try:
                        cache[key] = (
                            float(rel_col),
                            str(row.get("relevance_reasoning", "") or ""),
                        )
                        continue
                    except (TypeError, ValueError):
                        pass

                # Fall back to JSON for un-backfilled rows
                for h in qwen_predict.HORIZONS:
                    pred_json = row.get(f"pred_{h}")
                    if isinstance(pred_json, str) and pred_json.strip():
                        try:
                            pred = json.loads(pred_json)
                            rel = pred.get("relevance")
                            if rel is not None:
                                cache[key] = (
                                    float(rel),
                                    str(pred.get("relevance_reasoning", "")),
                                )
                                break
                        except Exception:
                            pass

    # 2. Own checkpoint cache (latest run wins)
    if os.path.exists(RELEVANCE_CACHE_PATH):
        try:
            df = pd.read_parquet(RELEVANCE_CACHE_PATH)
            for _, row in df.iterrows():
                key = (
                    str(row.get("ticker", "")).strip(),
                    str(row.get("date", ""))[:10],
                    str(row.get("text_snippet", ""))[:200],
                )
                cache[key] = (
                    float(row.get("relevance", 0.0)),
                    str(row.get("relevance_reasoning", "")),
                )
        except Exception:
            pass

    return cache


def save_relevance_cache(cache):
    rows = [
        {
            "ticker": ticker,
            "date": date_str,
            "text_snippet": snippet,
            "relevance": rel,
            "relevance_reasoning": reason,
        }
        for (ticker, date_str, snippet), (rel, reason) in cache.items()
    ]
    if not rows:
        return
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), RELEVANCE_CACHE_PATH)


def extract_relevance(ticker, article_text, date):
    """One factor-extraction LLM call. Returns (relevance, reasoning)."""
    company_context = qwen_predict.get_company_context(ticker)
    prompt = qwen_predict.build_factor_extraction_prompt(
        ticker, article_text, company_context, date
    )
    for _attempt in range(qwen_predict.MAX_RETRIES):
        raw = qwen_predict.call_llm(prompt)
        cleaned = qwen_predict.clean_raw_output(raw)
        parsed = qwen_predict.extract_json(cleaned)
        if isinstance(parsed, dict) and "relevance" in parsed:
            try:
                rel = max(0.0, min(1.0, float(parsed["relevance"])))
                reason = str(parsed.get("relevance_reasoning", ""))
                return rel, reason
            except (TypeError, ValueError):
                continue
    return 0.0, "extraction failed"


def score_all(candidates, cache):
    """Compute relevance for every candidate row, using cache where available.

    Returns the candidates df with two new columns: _relevance, _relevance_reasoning.
    """
    rels, reasons = [], []
    n_cached = 0
    n_scored_new = 0
    n_short = 0
    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        key = make_article_key(row)

        if key in cache:
            rel, reason = cache[key]
            rels.append(rel)
            reasons.append(reason)
            n_cached += 1
            continue

        article_text = qwen_predict.clean_article_text(row.get("Article_text", ""))
        if not article_text or len(article_text) < 150:
            rel, reason = 0.0, "article too short"
            rels.append(rel)
            reasons.append(reason)
            cache[key] = (rel, reason)
            n_short += 1
            continue

        # Date for the prompt (key already has its own normalised form)
        d = row.get("Date")
        if isinstance(d, (int, float)):
            date_str = pd.to_datetime(d, unit="ms").strftime("%Y-%m-%d")
        else:
            date_str = str(d)[:10]

        ticker = str(row.get("Stock_symbol", "")).strip()
        rel, reason = extract_relevance(ticker, article_text, date_str)
        rels.append(rel)
        reasons.append(reason)
        cache[key] = (rel, reason)
        n_scored_new += 1

        if n_scored_new % CHECKPOINT_EVERY == 0:
            save_relevance_cache(cache)
            print(
                f"  [{i}/{total}] cached={n_cached} new={n_scored_new} "
                f"short={n_short} (saved cache)"
            )

    save_relevance_cache(cache)
    print(f"  Done. cached={n_cached}, newly_scored={n_scored_new}, short={n_short}")

    df = candidates.copy()
    df["_relevance"] = rels
    df["_relevance_reasoning"] = reasons
    return df


# ============================================================
# Step 2.5: cap candidates per ticker (cached first, recent uncached after)
# ============================================================

def cap_per_ticker(candidates, cache, max_per_ticker):
    """Cap each ticker's candidate pool to max_per_ticker rows.

    Selection priority:
      1. Every row whose key is already in the relevance cache (free at scoring time)
      2. Most recent uncached rows, until the cap is reached.

    This minimises new LLM calls while still giving the relevance-ranking step
    a meaningfully-sized pool to pick the top 200 from.
    """
    if max_per_ticker is None or max_per_ticker <= 0:
        return candidates

    cache_keys = set(cache.keys())
    pieces = []

    for ticker in TARGET_TICKERS:
        sub = candidates[
            candidates["Stock_symbol"].astype(str).str.strip() == ticker
        ].copy()
        if len(sub) == 0:
            continue

        sub["_is_cached"] = sub.apply(
            lambda r: make_article_key(r) in cache_keys, axis=1
        )
        cached_rows = sub[sub["_is_cached"]]
        uncached_rows = sub[~sub["_is_cached"]]

        n_cached = len(cached_rows)
        n_remaining = max(0, max_per_ticker - n_cached)

        if n_remaining > 0 and len(uncached_rows) > 0:
            recent_uncached = uncached_rows.sort_values(
                "Date", ascending=False
            ).head(n_remaining)
        else:
            recent_uncached = uncached_rows.iloc[0:0]

        combined = pd.concat([cached_rows, recent_uncached], ignore_index=True)
        combined = combined.drop(columns=["_is_cached"])
        pieces.append(combined)

        print(
            f"  {ticker}: {n_cached} cached + "
            f"{len(recent_uncached)} most-recent uncached = {len(combined)} kept"
        )

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ============================================================
# Step 3: select top N per ticker
# ============================================================

def select_top_n(df, n_per_ticker):
    pieces = []
    for ticker in TARGET_TICKERS:
        sub = df[df["Stock_symbol"].astype(str).str.strip() == ticker].copy()
        if len(sub) == 0:
            print(f"  {ticker}: 0 candidates — skipping")
            continue
        sub = sub.sort_values("_relevance", ascending=False, kind="mergesort")
        n_avail = len(sub)
        n_take = min(n_per_ticker, n_avail)
        head = sub.head(n_take)
        median_rel = head["_relevance"].median()
        min_rel = head["_relevance"].min()
        print(
            f"  {ticker}: {n_avail} candidates -> top {n_take}  "
            f"(median rel = {median_rel:.2f}, min rel = {min_rel:.2f})"
        )
        pieces.append(head)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ============================================================
# Step 4: save per-ticker focused parquets
# ============================================================

def save_per_ticker(df, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for ticker in TARGET_TICKERS:
        sub = df[df["Stock_symbol"].astype(str).str.strip() == ticker]
        if len(sub) == 0:
            continue
        # Drop helper columns; qwen_predict only reads source columns.
        keep_cols = [c for c in sub.columns if not c.startswith("_")]
        out_path = os.path.join(target_dir, f"{ticker}.parquet")
        pq.write_table(pa.Table.from_pandas(sub[keep_cols]), out_path)
        print(f"  💾 {out_path} ({len(sub)} rows)")


# ============================================================
# Main
# ============================================================

def main():
    print(f"📋 Building focused dataset for {len(TARGET_TICKERS)} tickers")
    print(f"   Target: top {TOP_N_PER_TICKER} per ticker by relevance\n")

    print("Step 1/5: Collect candidates with 1d ground truth")
    candidates = collect_candidates()
    print(f"  Total candidates: {len(candidates)}")
    for ticker in TARGET_TICKERS:
        n = (candidates["Stock_symbol"].astype(str).str.strip() == ticker).sum()
        print(f"    {ticker}: {n}")

    print("\nStep 2/5: Load relevance cache")
    cache = load_relevance_cache()
    print(f"  Cached scores available for {len(cache)} articles")

    if MAX_CANDIDATES_PER_TICKER is not None:
        print(
            f"\nStep 3/5: Cap each ticker to {MAX_CANDIDATES_PER_TICKER} "
            f"candidates (cached first, recent uncached after)"
        )
        candidates = cap_per_ticker(candidates, cache, MAX_CANDIDATES_PER_TICKER)
        print(f"  After cap: {len(candidates)}")
    else:
        print("\nStep 3/5: No per-ticker cap — using all candidates")

    print("\nStep 4/5: Score relevance on uncached articles")
    scored = score_all(candidates, cache)

    print(f"\nStep 5/5: Select top {TOP_N_PER_TICKER} per ticker by relevance")
    selected = select_top_n(scored, TOP_N_PER_TICKER)
    print(f"  Total selected: {len(selected)}")

    print(f"\nWriting focused dataset to {FOCUSED_DIR}/")
    save_per_ticker(selected, FOCUSED_DIR)

    print("\n✅ Focused dataset built. Next steps:")
    print("   1. In qwen_predict.py:")
    print(f"        RETURN_DIR          = '{FOCUSED_DIR}'")
    print("        PRED_DIR            = 'data/focused_predictions_qwen_k5'")
    print("        K_SAMPLES           = 5")
    print("        RELEVANCE_THRESHOLD = 0.0   # already filtered upstream")
    print("   2. Run: python3 qwen_predict.py")
    print("   3. In merge_evaluation.py:")
    print("        PRED_DIR = 'data/focused_predictions_qwen_k5'")
    print("      then run it.")


if __name__ == "__main__":
    main()
