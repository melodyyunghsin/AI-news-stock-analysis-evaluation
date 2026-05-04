import requests
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import yfinance as yf
import os
import json
import time
import random
import re

# ============================================================
# CONFIG
# ============================================================

API_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    raise RuntimeError(
        "GEMINI_API_KEYS env var is not set. "
        "Export one or more comma-separated keys before running, e.g.\n"
        "    export GEMINI_API_KEYS=key1,key2,key3"
    )
MODEL   = "models/gemini-3.1-flash-lite-preview"
_call_count = 0

RETURN_DIR = "data/return_batches_clean"  # Use pre-cleaned articles
PRED_DIR   = "data/prediction_batches"
EVAL_DIR   = "data/evaluation_results"
PRICE_DIR  = "data/full_history"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

THROTTLE_SEC = 3  # Stay within Groq's 30 RPM free tier limit
MAX_RETRIES  = 3

HORIZONS = ["1d", "3d", "5d", "10d", "21d"]

# Toggle between enhanced two-step pipeline and legacy single-prompt pipeline
USE_ENHANCED_PIPELINE = True  # Set to False to use legacy single-prompt pipeline

# Module-level cache for company context (avoids redundant yfinance calls within a run)
_company_context_cache = {}

# ============================================================
# HELPERS
# ============================================================

def clean_raw_output(text):
    if not text:
        return None
    return (
        text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
    )


def extract_json(text):
    if not text:
        return None

    try:
        return json.loads(text)
    except:
        pass

    matches = re.findall(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    return None

def get_api_url():
    global _call_count
    key = API_KEYS[_call_count % len(API_KEYS)]
    _call_count += 1
    return f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={key}"

def call_llm(prompt):
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    while True:
        try:
            r = requests.post(get_api_url(), json=body, timeout=60)

            if r.status_code == 200:
                try:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except:
                    return None

            elif r.status_code == 429:
                wait = 4 + random.random() * 2
                print(f"⚠️ Rate limited, waiting {wait:.1f}s — {r.text}")
                time.sleep(wait)
                continue

            else:
                print("❌ API error:", r.status_code, r.text)  # add r.text
                time.sleep(2)

        except Exception as e:
            print("❌ Network error:", e)
            time.sleep(3)


def clean_article_text(text):
    """Strip Nasdaq FNSPID scraper junk prefix and trailing boilerplate.

    Articles scraped from nasdaq.com start with an error page message
    followed by date/author metadata, then 'Written by X for Source->'
    before the actual content begins. The end has Nasdaq disclaimers
    and session boilerplate.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    cleaned = text

    # Strip leading junk: find the "->" after "Written by...for...Source->"
    if "try using other words" in cleaned[:400] or "working diligently" in cleaned[:400]:
        arrow_idx = cleaned.find("->")
        if arrow_idx != -1 and arrow_idx < 600:
            cleaned = cleaned[arrow_idx + 2:]
        else:
            understanding_idx = cleaned.find("understanding.\n")
            if understanding_idx != -1 and understanding_idx < 400:
                cleaned = cleaned[understanding_idx + len("understanding.\n"):]

    # Strip secondary source lines
    cleaned = cleaned.lstrip("\n")
    source_prefixes = [
        "InvestorPlace - Stock Market News",
        "MarketBeat -",
        "ETF Trends -",
    ]
    for prefix in source_prefixes:
        if cleaned.startswith(prefix):
            newline_idx = cleaned.find("\n")
            if newline_idx != -1 and newline_idx < 200:
                cleaned = cleaned[newline_idx + 1:]

    # Strip trailing Nasdaq boilerplate
    cut_markers = [
        "The views and opinions expressed herein",
        "This data feed is not available",
        "\u00a9 2025, Nasdaq, Inc.",
        "\u00a9 2024, Nasdaq, Inc.",
        "\u00a9 2023, Nasdaq, Inc.",
        "To add symbols:",
        "Smart Portfolio is supported by our partner TipRanks",
    ]
    for marker in cut_markers:
        pos = cleaned.find(marker)
        if pos != -1:
            cleaned = cleaned[:pos]

    # Strip Motley Fool disclosures
    for marker in ["The Motley Fool has a disclosure policy.",
                   "The Motley Fool has positions in and recommends"]:
        pos = cleaned.find(marker)
        if pos != -1:
            cleaned = cleaned[:pos]

    return cleaned.strip()


# ============================================================
# COMPANY CONTEXT (yfinance)
# ============================================================

def get_company_context(ticker):
    """Fetch company name, sector, industry, and a brief description via yfinance.

    Results are cached in _company_context_cache so repeated calls for the
    same ticker within a run are free.  If yfinance fails for any reason the
    pipeline continues with a minimal fallback string.
    """
    if ticker in _company_context_cache:
        return _company_context_cache[ticker]

    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        summary = info.get("longBusinessSummary", "")
        if len(summary) > 300:
            summary = summary[:297] + "..."

        context = (
            f"Company: {name} ({ticker})\n"
            f"Sector: {sector} | Industry: {industry}\n"
            f"Description: {summary}"
        )
    except Exception as e:
        print(f"  ⚠️ yfinance lookup failed for {ticker}: {e}")
        context = f"{ticker} — company details unavailable."

    _company_context_cache[ticker] = context
    return context


# ============================================================
# PRICE SUMMARY (replaces raw price dump)
# ============================================================

def build_price_summary(ticker, article_date, days=60):
    """Build a concise, indicator-rich price summary instead of dumping raw prices.

    Research (Elahi & Taghvaei 2024) shows LLMs perform poorly on mental math
    over long number lists.  Pre-computing moving averages, trend direction,
    momentum, and volatility lets the model focus on interpretation.
    """
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return "Historical price data unavailable."

    try:
        df = pd.read_csv(path)
    except Exception:
        return "Historical price data unavailable."

    if "date" not in df.columns or "close" not in df.columns:
        return "Historical price data unavailable."

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    article_date = pd.to_datetime(article_date)

    hist = df[df["date"] < article_date].tail(days).copy()
    if len(hist) == 0:
        return "Historical price data unavailable."

    closes = hist["close"].astype(float).values
    dates = hist["date"].values
    n = len(closes)
    current_price = closes[-1]

    lines = [f"Price summary for {ticker} (as of {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}, {n} trading days of history):"]
    lines.append(f"Current price: ${current_price:.2f}")

    # --- Recent daily moves (last 10 days) ---
    recent_n = min(10, n)
    if recent_n >= 2:
        lines.append(f"\nRecent daily moves (last {recent_n} trading days):")
        start_idx = n - recent_n
        for i in range(max(start_idx, 1), n):
            pct = (closes[i] - closes[i - 1]) / closes[i - 1] * 100
            arrow = "↑" if pct >= 0 else "↓"
            d = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
            lines.append(f"  {d}: {arrow} {abs(pct):.2f}%")

    # --- Trend summaries ---
    def pct_change_over(period):
        if n > period:
            old = closes[-(period + 1)]
            return (current_price - old) / old * 100
        return None

    for label, period in [("5-day", 5), ("20-day", 20), ("60-day", 60)]:
        pct = pct_change_over(period)
        if pct is not None:
            direction = "up" if pct >= 0 else "down"
            lines.append(f"{label} trend: {direction} {abs(pct):.2f}%")

    # --- Simple Moving Averages ---
    sma_section = []
    for window in [10, 20, 50]:
        if n >= window:
            sma = float(np.mean(closes[-window:]))
            position = "ABOVE" if current_price >= sma else "BELOW"
            sma_section.append(f"  {window}-day SMA: ${sma:.2f} (price is {position})")
    if sma_section:
        lines.append("\nMoving averages:")
        lines.extend(sma_section)

    # --- Volatility ---
    if n >= 21:
        daily_abs_pct = [abs(closes[i] - closes[i - 1]) / closes[i - 1] * 100 for i in range(n - 20, n)]
        avg_vol = np.mean(daily_abs_pct)
        lines.append(f"\n20-day average absolute daily move (volatility): {avg_vol:.2f}%")

    # --- Recent range ---
    if n >= 20:
        high_20 = float(np.max(closes[-20:]))
        low_20 = float(np.min(closes[-20:]))
        lines.append(f"20-day closing range: ${low_20:.2f} – ${high_20:.2f}")

    # --- Volume comparison (if column exists) ---
    has_volume = "volume" in hist.columns
    if has_volume:
        vols = hist["volume"].astype(float).values
        if n >= 10:
            avg_vol_10 = float(np.mean(vols[-10:]))
            avg_vol_60 = float(np.mean(vols)) if n >= 60 else float(np.mean(vols))
            ratio = avg_vol_10 / avg_vol_60 if avg_vol_60 > 0 else 1.0
            vol_note = "elevated" if ratio > 1.15 else ("low" if ratio < 0.85 else "normal")
            lines.append(f"\nVolume: 10-day avg {avg_vol_10:,.0f} vs {n}-day avg {avg_vol_60:,.0f} ({vol_note})")

    return "\n".join(lines)


# ============================================================
# FACTOR EXTRACTION PROMPT (Step 1 of two-step pipeline)
# ============================================================

def build_factor_extraction_prompt(ticker, text, company_context, date):
    """Build prompt for the first LLM call: extract causal factors from the article.

    Inspired by LLMFactor (Wang et al., ACL 2024) whose ablation showed that
    explicit factor extraction contributed ~9% accuracy and ~46% of total MCC
    improvement over direct prediction.
    """
    text = text.strip()[:5000]

    return f"""You are an expert financial analyst. Your task is to extract specific causal factors from the article below that could affect the stock price of {ticker}.

{company_context}

Temporal restriction: Pretend today is {date}. Use ONLY information that would be known on or before that date. NO hindsight.

Article:
\"\"\"{text}\"\"\"

Instructions:
- Identify 3 to 5 specific factors from this article that could influence {ticker}'s stock price.
- For each factor, provide:
  - "factor": a short description of the causal factor (1-2 sentences)
  - "direction": "positive" or "negative" (the expected impact on {ticker}'s stock price)
  - "time_horizon": "short-term" (1-5 days), "medium-term" (1-4 weeks), or "long-term" (months+)
  - "confidence": "high", "medium", or "low"
- If the article is NOT directly about {ticker}, still identify indirect effects (industry trends, competitor news, macro factors) but mark confidence as "low" or "medium".
- Consider supply chain effects, competitive dynamics, regulatory implications, and market sentiment.

You MUST output ONLY a valid JSON array. NO markdown. NO code fences. NO commentary before or after the JSON.

Example format:
[
  {{"factor": "...", "direction": "positive", "time_horizon": "short-term", "confidence": "high"}},
  {{"factor": "...", "direction": "negative", "time_horizon": "medium-term", "confidence": "medium"}}
]"""


def format_factors_for_prediction(factors_json):
    """Format extracted factors into readable text for the prediction prompt."""
    if not factors_json or not isinstance(factors_json, list):
        return "Factor extraction failed. Analyze the article directly."

    lines = ["Extracted causal factors:"]
    for i, f in enumerate(factors_json, 1):
        factor = f.get("factor", "N/A")
        direction = f.get("direction", "N/A")
        horizon = f.get("time_horizon", "N/A")
        confidence = f.get("confidence", "N/A")
        lines.append(f"  {i}. {factor}")
        lines.append(f"     Impact: {direction} | Horizon: {horizon} | Confidence: {confidence}")
    return "\n".join(lines)


# ============================================================
# PREDICTION PROMPT (Step 2 of two-step pipeline)
# ============================================================

# Horizon-specific instructions informed by FinGPT (Liang et al., 2024):
# differentiating short-term vs long-term news impact improved accuracy ~4pp.
HORIZON_INSTRUCTIONS = {
    "1d": (
        "Focus on immediate market reaction. Consider whether this news is likely "
        "already priced in. Short-term sentiment and momentum dominate."
    ),
    "3d": (
        "Consider both immediate reaction and follow-on effects. Weigh whether the "
        "identified factors have short-term or medium-term implications."
    ),
    "5d": (
        "Consider both immediate reaction and follow-on effects. Weigh whether the "
        "identified factors have short-term or medium-term implications."
    ),
    "10d": (
        "Focus on structural and fundamental impacts. Consider industry dynamics, "
        "competitive positioning, and whether this news changes the medium-term "
        "outlook. Short-term noise is less relevant."
    ),
    "21d": (
        "Focus on structural and fundamental impacts. Consider industry dynamics, "
        "competitive positioning, and whether this news changes the medium-term "
        "outlook. Short-term noise is less relevant."
    ),
}


def build_prediction_prompt(article_id, date, ticker, text, horizon,
                            price_summary, company_context, factors_text):
    """Build prompt for the second LLM call: predict direction using extracted factors.

    The prompt integrates company context, pre-computed price indicators, and
    the causal factors from step 1, with horizon-specific reasoning guidance
    (FinGPT, Liang et al., 2024).
    """
    text = text.strip()[:3000]
    horizon_instruction = HORIZON_INSTRUCTIONS.get(horizon, "")

    return f"""You are an expert financial analyst predicting stock price direction for {ticker}.

{company_context}

{factors_text}

{price_summary}

Horizon-specific guidance ({horizon}):
{horizon_instruction}

Original article (for reference):
\"\"\"{text}\"\"\"

Based on the causal factors above, the price data, and the article, predict whether {ticker} will move UP or DOWN over the next {horizon}.

Temporal restriction: Pretend you are predicting at {date}. Use ONLY information known before or at that date. NO hindsight.

You MUST output ONLY valid JSON. NO markdown. NO code fences. NO extra text before or after.

Return EXACTLY this structure:

{{
  "article_id": "{article_id}",
  "ticker": "{ticker}",
  "direction": "UP" | "DOWN",
  "strength": "weak" | "moderate" | "strong",
  "expected_move_percent": number,
  "relevance": "high" | "medium" | "low",
  "explanation": "short explanation"
}}

Rules:
- direction MUST be either "UP" or "DOWN" — never omit a direction
- Even if the news seems mixed or loosely related, commit to whichever direction is more likely
- Stocks move every day; your job is to predict which direction this news tips the balance
- Predict ONLY {ticker} over a {horizon} horizon
- Strength based on expected_move_percent:
  - <= 1% → weak
  - 1% < x <= 5% → moderate
  - > 5% → strong
- expected_move_percent must be a plain number (e.g. 1.5, not "1.5%") and MUST be > 0
- relevance: "high" if the article is directly about {ticker} or its core business, "medium" if about the sector/competitors/supply chain, "low" if only tangentially related
- ALWAYS include all fields including "relevance"
- Do NOT mention other companies in the explanation"""


# ============================================================
# LEGACY FUNCTIONS (kept for A/B testing)
# ============================================================

def load_historical_prices_legacy(ticker, article_date, days=60):
    """Legacy: returns raw price list. Kept for A/B comparison."""
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return "Historical prices unavailable."

    try:
        df = pd.read_csv(path)
    except Exception:
        return "Historical prices unavailable."

    if "date" not in df.columns or "close" not in df.columns:
        return "Historical prices unavailable."

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    article_date = pd.to_datetime(article_date)

    mask = df["date"] < article_date
    historical = df[mask].tail(days)

    if len(historical) == 0:
        return "Historical prices unavailable."

    hist_lines = [f"{row['date'].strftime('%Y-%m-%d')}: {float(row['close'])}" for _, row in historical.iterrows()]

    closes = historical["close"].astype(float).values
    if len(closes) >= 2:
        daily_moves = [abs(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
        avg_daily_move = sum(daily_moves) / len(daily_moves)
        vol_line = f"Average daily price move (volatility): {avg_daily_move:.2f}%"
    else:
        vol_line = ""

    header = f"Historical prices (last {len(historical)} trading days before {article_date.strftime('%Y-%m-%d')}):\n"
    return header + "\n".join(hist_lines) + (f"\n{vol_line}" if vol_line else "")


def build_prompt_legacy(article_id, date, ticker, text, horizon, hist_text):
    """Legacy: single-prompt prediction. Kept for A/B comparison."""
    text = text.strip().replace('"', '\\"')[:5000]

    return f"""
You are an expert financial analyst predicting short-term stock price direction for {ticker}.
Given the article below, predict whether {ticker} will move UP or DOWN over the next {horizon}.

You MUST output ONLY valid JSON.
NO markdown. NO code fences. NO extra text.

Return EXACTLY this structure:

{{
  "article_id": "{article_id}",
  "ticker": "{ticker}",
  "direction": "UP" | "DOWN",
  "strength": "weak" | "moderate" | "strong",
  "expected_move_percent": number,
  "explanation": "short explanation"
}}

Rules:
- direction MUST be either "UP" or "DOWN" — never omit a direction
- Even if the news seems mixed or loosely related, commit to whichever direction is more likely
- Stocks move every day; your job is to predict which direction this news tips the balance
- Predict ONLY {ticker} over a {horizon} horizon
- Strength based on expected_move_percent:
  - <= 1% → weak
  - 1% < x <= 5% → moderate
  - > 5% → strong
- expected_move_percent must be a plain number (e.g. 1.5, not "1.5%") and MUST be > 0
- No hindsight bias — treat the article date as "now"
- ALWAYS include all fields
- Do NOT mention other companies

Temporal restriction:
Pretend you are predicting at {date}.
Use ONLY information known before or at that date.
NO hindsight.

{hist_text}

Article:
\"\"\"{text}\"\"\"
"""


def predict_for_row_legacy(row, hist_text):
    """Legacy: single-prompt prediction for all horizons. Kept for A/B comparison."""
    predictions = {}

    article_text = clean_article_text(row.get("Article_text", ""))
    if len(article_text) < 150:
        return predictions

    article_id = str(row.get("Article_id", ""))
    ticker = str(row.get("Stock_symbol", "")).strip()
    date = str(row.get("Date", ""))[:10]

    print(f"  [legacy] Predicting for {ticker} at {date}")

    for horizon in HORIZONS:
        print(f"    {horizon}...")

        prompt = build_prompt_legacy(article_id, date, ticker, article_text, horizon, hist_text)

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(THROTTLE_SEC)

            raw = call_llm(prompt)
            clean = clean_raw_output(raw)
            parsed = extract_json(clean)

            if isinstance(parsed, dict):
                break

            print(f"      ⚠️ Parse failed (attempt {attempt})")

        if parsed is None:
            predictions[horizon] = None
        else:
            predictions[horizon] = parsed

    return predictions


# ============================================================
# ENHANCED TWO-STEP PREDICTION
# ============================================================

def predict_for_row(row):
    """Two-step prediction: factor extraction then horizon-specific prediction.

    Step 1 (shared): Extract causal factors from the article (1 API call).
    Step 2 (per-horizon): Predict direction using factors + price data + horizon
           guidance (5 API calls).
    Total: 6 API calls per row instead of the legacy 5, but with much richer
    context in each prediction call.
    """
    predictions = {}

    article_text = clean_article_text(row.get("Article_text", ""))
    if len(article_text) < 150:
        return predictions

    article_id = str(row.get("Article_id", ""))
    ticker = str(row.get("Stock_symbol", "")).strip()
    date = str(row.get("Date", ""))[:10]

    print(f"  Predicting for {ticker} at {date}")

    # 1. Company context (cached)
    company_context = get_company_context(ticker)

    # 2. Price summary (once per row)
    price_summary = build_price_summary(ticker, date)

    # 3. Factor extraction (once per row — factors are horizon-independent)
    print(f"    Extracting factors...")
    factor_prompt = build_factor_extraction_prompt(ticker, article_text, company_context, date)

    factors_json = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(THROTTLE_SEC)
        raw = call_llm(factor_prompt)
        clean = clean_raw_output(raw)
        parsed = extract_json(clean)
        if isinstance(parsed, list):
            factors_json = parsed
            break
        print(f"      ⚠️ Factor extraction parse failed (attempt {attempt})")

    if factors_json is None:
        print(f"    ⚠️ Factor extraction failed after {MAX_RETRIES} attempts, proceeding without factors")

    factors_text = format_factors_for_prediction(factors_json)

    # 4. Per-horizon predictions
    for horizon in HORIZONS:
        print(f"    {horizon}...")

        prompt = build_prediction_prompt(
            article_id, date, ticker, article_text, horizon,
            price_summary, company_context, factors_text,
        )

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(THROTTLE_SEC)
            raw = call_llm(prompt)
            clean = clean_raw_output(raw)
            parsed = extract_json(clean)
            if isinstance(parsed, dict):
                break
            print(f"      ⚠️ Parse failed (attempt {attempt})")

        predictions[horizon] = parsed

    return predictions


# ============================================================
# FILE PROCESSING & MAIN
# ============================================================

def process_file(return_path):
    print(f"📈 Processing: {os.path.basename(return_path)}")
    pipeline_label = "enhanced" if USE_ENHANCED_PIPELINE else "legacy"
    print(f"  Pipeline: {pipeline_label}")

    # Resume support: if a partial prediction file already exists, load it
    # so we keep previously completed rows and only process remaining ones.
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if os.path.exists(pred_path):
        print(f"  ♻️ Found existing predictions, resuming from checkpoint...")
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_parquet(return_path)

    # Ensure prediction columns exist
    for horizon in HORIZONS:
        if f"pred_{horizon}" not in df.columns:
            df[f"pred_{horizon}"] = None

    total_rows = len(df)
    skipped = 0
    processed = 0

    for idx in range(total_rows):
        row = df.iloc[idx]

        # Skip if no article text
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue

        # Skip rows that already have all horizons predicted
        all_done = all(
            row.get(f"pred_{h}") is not None
            and isinstance(row.get(f"pred_{h}"), str)
            and row.get(f"pred_{h}").strip() != ""
            for h in HORIZONS
        )
        if all_done:
            skipped += 1
            continue

        ticker = str(row.get("Stock_symbol", "")).strip()
        date = str(row.get("Date", ""))[:10]

        if USE_ENHANCED_PIPELINE:
            predictions = predict_for_row(row)
        else:
            hist_text = load_historical_prices_legacy(ticker, date)
            predictions = predict_for_row_legacy(row, hist_text)

        # Store predictions
        for horizon, pred in predictions.items():
            if pred is not None:
                df.at[idx, f"pred_{horizon}"] = json.dumps(pred)

        processed += 1
        print(f"  ✅ Processed {processed} new (skipped {skipped} already done) / {total_rows} total")

        # Save checkpoint after each row so progress is not lost on interruption
        pq.write_table(pa.Table.from_pandas(df), pred_path)

    if skipped > 0:
        print(f"  ♻️ Skipped {skipped} rows that were already predicted")

    # Final save
    pq.write_table(pa.Table.from_pandas(df), pred_path)
    print(f"  💾 Saved predictions to: {pred_path}")

    # Create evaluation CSV
    eval_rows = []

    for _, row in df.iterrows():
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue

        eval_row = {
            "Article_id": row.get("Article_id"),
            "Date": row.get("Date"),
            "Ticker": row.get("Stock_symbol"),
        }

        # Add predictions for each horizon
        for horizon in HORIZONS:
            pred_json = row.get(f"pred_{horizon}")
            if pred_json and isinstance(pred_json, str):
                try:
                    pred = json.loads(pred_json)
                    eval_row[f"Pred_Direction_{horizon}"] = pred.get("direction")
                    eval_row[f"Pred_Strength_{horizon}"] = pred.get("strength")
                    eval_row[f"Pred_Move_{horizon}"] = pred.get("expected_move_percent")
                    eval_row[f"Pred_Relevance_{horizon}"] = pred.get("relevance")
                    eval_row[f"Explanation_{horizon}"] = pred.get("explanation")
                except:
                    eval_row[f"Pred_Direction_{horizon}"] = None
                    eval_row[f"Pred_Strength_{horizon}"] = None
                    eval_row[f"Pred_Move_{horizon}"] = None
                    eval_row[f"Pred_Relevance_{horizon}"] = None
                    eval_row[f"Explanation_{horizon}"] = None
            else:
                eval_row[f"Pred_Direction_{horizon}"] = None
                eval_row[f"Pred_Strength_{horizon}"] = None
                eval_row[f"Pred_Move_{horizon}"] = None
                eval_row[f"Pred_Relevance_{horizon}"] = None
                eval_row[f"Explanation_{horizon}"] = None

        # Add actual returns for each horizon
        for horizon in HORIZONS:
            eval_row[f"Actual_Return_{horizon}"] = row.get(f"return_{horizon}")
            eval_row[f"Actual_Direction_{horizon}"] = row.get(f"actual_direction_{horizon}")
            eval_row[f"Actual_Strength_{horizon}"] = row.get(f"actual_strength_{horizon}")

        eval_rows.append(eval_row)

    # Save evaluation CSV
    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(EVAL_DIR, os.path.basename(return_path).replace(".parquet", "_eval.csv"))
    eval_df.to_csv(eval_path, index=False)
    print(f"  📄 Saved evaluation to: {eval_path}")


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No return batch parquet files found.")
        return

    for f in files:
        return_path = os.path.join(RETURN_DIR, f)
        process_file(return_path)


if __name__ == "__main__":
    main()
