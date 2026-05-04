"""Claude API ceiling test for the news -> stock direction pipeline.

Runs a sample of articles through the Anthropic API using the same two-step
prompt structure as qwen_predict.py (factor extraction + per-horizon prediction)
so we can compare a frontier model directly against the local Qwen-7B baseline
and against the always-UP majority class.

Question this answers:
    Is the local 7B model the bottleneck, or does the data lack signal at
    these horizons?

Outputs:
    data/claude_ceiling_test/claude_predictions.parquet
    data/claude_ceiling_test/comparison_summary.csv

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python3 claude_ceiling_test.py

Cost estimate (claude-opus-4-7, ~100 articles, 6 calls each, ~3K avg input tokens):
    Input:  ~1.8M tokens * $5/M  = ~$9
    Output: ~0.18M tokens * $25/M = ~$5
    Total:  ~$10-15
"""

import os
import sys
import json
import random

import pandas as pd
import anthropic

import qwen_predict  # reuse prompt builders, cleaners, helpers

# ============================================================
# CONFIG
# ============================================================

# claude-opus-4-7 is the most capable Claude model. To cut cost ~3x while
# still using a frontier model, change to "claude-sonnet-4-6".
MODEL = "claude-opus-4-7"

SAMPLE_SIZE = 100  # articles

# Restrict to the highest-priority tickers so the comparison rests on names
# Qwen has many predictions for and the model has plenty of training context on.
PRIORITY_TICKERS = set(qwen_predict.TICKER_PRIORITY[:10])

PRED_DIR = "data/prediction_batches_qwen"
OUTPUT_DIR = "data/claude_ceiling_test"
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "claude_predictions.parquet")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZONS = qwen_predict.HORIZONS

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY env var not set. Export it before running:")
    print("    export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

client = anthropic.Anthropic(max_retries=5)


# ============================================================
# Claude API call (streaming for safety on large max_tokens)
# ============================================================

def call_claude(prompt_content):
    """Send a prompt to Claude and return (text, usage).

    `prompt_content` is either a string or a list of content blocks (the latter
    lets us add cache_control for prompt caching across horizon calls).
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt_content}],
    ) as stream:
        message = stream.get_final_message()

    text = next((b.text for b in message.content if b.type == "text"), "")
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
    }
    return text, usage


# ============================================================
# Prediction prompt — split for prompt caching across horizons
# ============================================================

def build_claude_prediction_prompt(article_id, date, ticker, text, horizon,
                                   price_summary, company_context, factors_text,
                                   relevance, relevance_reasoning):
    """Same content as qwen_predict.build_prediction_prompt, restructured so
    horizon-specific text is at the end. The shared prefix can be cache_control
    flagged for reuse across the 5 horizon calls of one article.

    Note: the cached prefix may fall under Opus 4.7's 4096-token minimum and
    silently not activate. Sonnet 4.6 has a 2048-token minimum and will cache.
    """
    text = text.strip()[:1500]  # match qwen_predict truncation for fair comparison
    horizon_instruction = qwen_predict.HORIZON_INSTRUCTIONS.get(horizon, "")

    cached_prefix = f"""You are an expert financial analyst predicting stock price direction for {ticker}.

{company_context}

Article relevance to {ticker}: {relevance:.2f} - {relevance_reasoning}

{factors_text}

{price_summary}

Original article (for reference):
\"\"\"{text}\"\"\"

Temporal restriction: Pretend you are predicting at {date}. Use ONLY information known before or at that date. NO hindsight.

You will be asked to predict stock direction over a specific horizon. Output ONLY a valid JSON object in this exact structure:

{{
  "article_id": "{article_id}",
  "ticker": "{ticker}",
  "direction": "UP" | "DOWN",
  "confidence": number between 0.0 and 1.0,
  "explanation": "short explanation"
}}

Rules:
- direction MUST be either "UP" or "DOWN"
- Even if the news seems mixed or loosely related, commit to whichever direction is more likely
- confidence: 0.0-1.0, where 0.8-1.0 = strong, 0.6-0.8 = moderate, 0.4-0.6 = low, <0.4 = essentially guessing
- confidence is about DIRECTION certainty, NOT price magnitude
- explanation: 1-2 sentences justifying the direction
- ALWAYS include all 4 fields. Output ONLY the JSON, no markdown, no code fences."""

    volatile_suffix = f"""

Now predict whether {ticker} will move UP or DOWN over the next {horizon}.

Horizon-specific guidance ({horizon}):
{horizon_instruction}"""

    return cached_prefix, volatile_suffix


# ============================================================
# Article sampling
# ============================================================

def sample_articles(existing_ids):
    """Pick eligible articles from the prediction parquets."""
    if not os.path.isdir(PRED_DIR):
        raise RuntimeError(
            f"Prediction dir not found: {PRED_DIR} - run qwen_predict.py first."
        )

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".parquet"))
    eligible = []
    for f in files:
        df = pd.read_parquet(os.path.join(PRED_DIR, f))
        for _, row in df.iterrows():
            ticker = str(row.get("Stock_symbol", "")).strip()
            if ticker not in PRIORITY_TICKERS:
                continue

            article_id = str(row.get("Article_id", ""))
            if article_id in existing_ids:
                continue

            article_text = qwen_predict.clean_article_text(row.get("Article_text", ""))
            if len(article_text) < 200:
                continue

            # Need ground truth for at least the 1d horizon
            if pd.isna(row.get("actual_direction_1d")):
                continue

            # Qwen must have a real (non-SKIPPED) prediction so the comparison
            # is apples to apples on the same article set.
            qwen_raw = row.get("pred_1d")
            if not isinstance(qwen_raw, str) or not qwen_raw.strip():
                continue
            try:
                if json.loads(qwen_raw).get("direction") == "SKIPPED":
                    continue
            except Exception:
                continue

            eligible.append({
                "article_id": article_id,
                "ticker": ticker,
                "date": str(row.get("Date", ""))[:10],
                "article_text": article_text,
                "qwen_predictions": {h: row.get(f"pred_{h}") for h in HORIZONS},
                "actual_directions": {h: row.get(f"actual_direction_{h}") for h in HORIZONS},
                "actual_returns": {h: row.get(f"return_{h}") for h in HORIZONS},
            })

    if not eligible:
        raise RuntimeError(
            "No eligible articles found. Ensure qwen_predict.py has produced "
            "non-SKIPPED predictions on the priority tickers with ground truth."
        )

    print(f"Found {len(eligible)} eligible articles across top "
          f"{len(PRIORITY_TICKERS)} tickers.")
    random.seed(42)
    random.shuffle(eligible)
    return eligible[:SAMPLE_SIZE]


# ============================================================
# Per-article prediction
# ============================================================

def predict_one_article(article):
    """Run factor extraction + 5 horizon predictions through Claude."""
    ticker = article["ticker"]
    date = article["date"]
    article_text = article["article_text"]
    article_id = article["article_id"]

    company_context = qwen_predict.get_company_context(ticker)

    # Step 1: factor extraction (no caching — single call per article)
    factor_prompt = qwen_predict.build_factor_extraction_prompt(
        ticker, article_text, company_context, date
    )
    raw, _ = call_claude(factor_prompt)
    parsed = qwen_predict.extract_json(qwen_predict.clean_raw_output(raw))

    if isinstance(parsed, dict) and "factors" in parsed:
        factors_data = parsed
        relevance = max(0.0, min(1.0, float(parsed.get("relevance", 0.0))))
        relevance_reasoning = str(parsed.get("relevance_reasoning", ""))
    else:
        factors_data = {"factors": []}
        relevance = 0.5
        relevance_reasoning = "factor extraction parse failed"

    factors_text = qwen_predict.format_factors_for_prediction(factors_data)
    price_summary = qwen_predict.build_price_summary(ticker, date)

    # Step 2: per-horizon predictions, with prompt caching across the 5 calls
    predictions = {}
    cache_reads_total = 0
    for horizon in HORIZONS:
        prefix, suffix = build_claude_prediction_prompt(
            article_id, date, ticker, article_text, horizon,
            price_summary, company_context, factors_text,
            relevance, relevance_reasoning,
        )
        content = [
            {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": suffix},
        ]
        raw, usage = call_claude(content)
        cache_reads_total += usage["cache_read_input_tokens"]
        parsed = qwen_predict.extract_json(qwen_predict.clean_raw_output(raw))

        if isinstance(parsed, dict) and parsed.get("direction") in ("UP", "DOWN"):
            predictions[horizon] = {
                "direction": parsed["direction"],
                "confidence": parsed.get("confidence"),
                "explanation": parsed.get("explanation", ""),
            }
        else:
            predictions[horizon] = None

    return {
        "relevance": relevance,
        "relevance_reasoning": relevance_reasoning,
        "predictions": predictions,
        "cache_reads_total": cache_reads_total,
    }


def article_to_rows(article, claude_result):
    """Flatten one article result into 5 per-horizon rows for the output df."""
    rows = []
    for horizon in HORIZONS:
        qwen_dir = None
        qwen_raw = article["qwen_predictions"].get(horizon)
        if isinstance(qwen_raw, str):
            try:
                qwen_dir = json.loads(qwen_raw).get("direction")
            except Exception:
                pass

        claude_pred = claude_result["predictions"].get(horizon) or {}
        rows.append({
            "Article_id": article["article_id"],
            "Ticker": article["ticker"],
            "Date": article["date"],
            "horizon": horizon,
            "Claude_Direction": claude_pred.get("direction"),
            "Claude_Confidence": claude_pred.get("confidence"),
            "Claude_Explanation": claude_pred.get("explanation"),
            "Qwen_Direction": qwen_dir,
            "Actual_Direction": article["actual_directions"].get(horizon),
            "Actual_Return": article["actual_returns"].get(horizon),
            "Claude_Relevance": claude_result["relevance"],
        })
    return rows


# ============================================================
# Comparison metrics
# ============================================================

def compute_summary(df):
    """Build per-horizon comparison: Claude vs Qwen vs always_UP."""
    rows = []
    for horizon in HORIZONS:
        sub = df[df["horizon"] == horizon].copy()
        sub = sub.dropna(subset=["Actual_Direction", "Claude_Direction", "Qwen_Direction"])
        sub = sub[sub["Actual_Direction"].astype(str).str.upper().isin(("UP", "DOWN"))]
        sub = sub[sub["Claude_Direction"].astype(str).str.upper().isin(("UP", "DOWN"))]
        sub = sub[sub["Qwen_Direction"].astype(str).str.upper().isin(("UP", "DOWN"))]
        n = len(sub)
        if n == 0:
            continue

        actual = sub["Actual_Direction"].astype(str).str.upper()
        claude = sub["Claude_Direction"].astype(str).str.upper()
        qwen = sub["Qwen_Direction"].astype(str).str.upper()

        rows.append({
            "horizon": horizon,
            "n": n,
            "claude_acc": float((claude == actual).mean()),
            "qwen_acc": float((qwen == actual).mean()),
            "always_up_acc": float((actual == "UP").mean()),
        })

    summary = pd.DataFrame(rows)
    if len(summary):
        summary["claude_vs_always_up"] = summary["claude_acc"] - summary["always_up_acc"]
        summary["claude_vs_qwen"] = summary["claude_acc"] - summary["qwen_acc"]
    return summary


# ============================================================
# Main
# ============================================================

def main():
    print(f"Claude ceiling test")
    print(f"  Model:       {MODEL}")
    print(f"  Sample size: {SAMPLE_SIZE}")
    print(f"  Tickers:     {sorted(PRIORITY_TICKERS)}\n")

    # Resume support
    existing_ids = set()
    saved_rows = []
    if os.path.exists(OUTPUT_PARQUET):
        prior = pd.read_parquet(OUTPUT_PARQUET)
        existing_ids = set(prior["Article_id"].astype(str).tolist())
        saved_rows = prior.to_dict("records")
        print(f"Resuming - {len(existing_ids)} article ids already done.\n")

    articles = sample_articles(existing_ids)
    print(f"Predicting {len(articles)} new articles.\n")

    all_rows = list(saved_rows)
    for i, article in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {article['ticker']} {article['date']}", flush=True)
        try:
            result = predict_one_article(article)
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error: {e}")
            continue

        all_rows.extend(article_to_rows(article, result))
        # Checkpoint after every article so a crash doesn't lose progress
        pd.DataFrame(all_rows).to_parquet(OUTPUT_PARQUET, index=False)

    if not all_rows:
        print("No predictions saved.")
        return

    df = pd.DataFrame(all_rows)
    summary = compute_summary(df)

    print("\n" + "=" * 78)
    print("Side-by-side comparison")
    print("=" * 78)
    if len(summary):
        print(f"  {'horizon':>7}  {'n':>4}  {'claude':>7}  {'qwen':>7}  "
              f"{'always_UP':>10}  {'C-UP':>7}  {'C-Q':>7}")
        for _, r in summary.iterrows():
            print(f"  {r['horizon']:>7}  {int(r['n']):>4}  "
                  f"{r['claude_acc']:>7.3f}  {r['qwen_acc']:>7.3f}  "
                  f"{r['always_up_acc']:>10.3f}  "
                  f"{r['claude_vs_always_up']:>+7.3f}  "
                  f"{r['claude_vs_qwen']:>+7.3f}")

    summary.to_csv(SUMMARY_CSV, index=False)

    # Verdict
    print("\nVerdict:")
    if not len(summary):
        print("  Insufficient data.")
        return

    n_horizons = len(summary)
    beats_majority = int((summary["claude_vs_always_up"] > 0.02).sum())
    beats_qwen = int((summary["claude_vs_qwen"] > 0.02).sum())

    if beats_majority >= 3:
        print(f"  Claude beats always_UP by >=2pp on {beats_majority}/{n_horizons} horizons.")
        print(f"  -> The local Qwen-7B model is the bottleneck.")
        print(f"     Worth scaling up (Qwen-32B locally, or run the full pipeline through Claude).")
    elif beats_majority >= 1:
        print(f"  Claude beats always_UP on {beats_majority}/{n_horizons} horizons (marginal).")
        print(f"  -> Some signal, but reformulating the task is more promising than scaling the model.")
        print(f"     Consider: abnormal returns vs SPY, BIG_UP/FLAT/BIG_DOWN buckets, pairwise comparison.")
    else:
        print(f"  Claude does NOT beat always_UP on any horizon.")
        print(f"  -> The data does not have predictable direction signal at these horizons.")
        print(f"     This is a real finding worth reporting. Consider:")
        print(f"       1. Reformulate the target (abnormal returns, magnitude buckets, pairwise)")
        print(f"       2. Acknowledge this as a methodological cautionary finding")

    print(f"\n  Claude vs Qwen: Claude better on {beats_qwen}/{n_horizons} horizons.")
    print(f"\nWrote: {OUTPUT_PARQUET}")
    print(f"Wrote: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
