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

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_URL  = f"{OLLAMA_BASE_URL}/chat/completions"
MODEL           = "qwen2.5:7b"

RETURN_DIR = "data/return_batches_clean"  # Use pre-cleaned articles
PRED_DIR   = "data/prediction_batches_qwen"
EVAL_DIR   = "data/evaluation_results_qwen"
PRICE_DIR  = "data/full_history"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

THROTTLE_SEC = 0  # Local Ollama — no rate limit
MAX_RETRIES  = 3

HORIZONS = ["1d", "3d", "5d", "10d", "21d"]

RELEVANCE_THRESHOLD = 0.4  # Skip prediction for articles below this relevance

# Self-consistency sampling for the prediction step.
# K_SAMPLES = 1: legacy behavior (one call per horizon, model self-rates confidence).
# K_SAMPLES > 1: call k times at SAMPLE_TEMPERATURE, majority-vote the direction,
#                use vote agreement (e.g. 4/5 = 0.8) as the calibrated confidence.
# K=5 typically adds 3-5pp accuracy and produces reliable calibration, at 5x cost.
K_SAMPLES = 1
SAMPLE_TEMPERATURE = 0.8

# Toggle between enhanced two-step pipeline and legacy single-prompt pipeline
USE_ENHANCED_PIPELINE = True  # Set to False to use legacy single-prompt pipeline

# Tickers are predicted in this order across ALL batches. If you stop the run
# partway through, the highest-priority tickers will be fully covered globally
# rather than partially covered across every ticker.
TICKER_PRIORITY = [
    # Tier 1: mega-cap, highest article volume + retail interest
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "GOOG", "AVGO", "TSM",
    # Tier 2: major semis + classic enterprise
    "AMD", "INTC", "QCOM", "ORCL", "CRM", "ADBE", "CSCO",
    # Tier 3: other large-cap (semi equipment, autos, industrials)
    "ASML", "MU", "TXN", "AMAT", "LRCX", "KLAC",
    "F", "GM", "TM", "BA", "GE", "CAT", "DE", "MMM",
    # Tier 4: mid-cap tech / auto / industrial
    "NOW", "PANW", "CRWD", "SNOW", "ZS", "OKTA", "TEAM",
    "DELL", "HPE", "HPQ", "NTAP", "SMCI",
    "NXPI", "ADI", "MRVL", "ON", "NIO", "HMC",
    "EMR", "ROK", "GD",
    # Tier 5: smaller / less liquid / less news coverage
    "JBL", "LOGI", "SWKS", "AAOI",
    "ERIC", "NOK", "BB"
]
_TICKER_RANK = {t: i for i, t in enumerate(TICKER_PRIORITY)}


def ticker_rank(ticker):
    """Lower rank = higher priority. Unknown tickers go to the end."""
    return _TICKER_RANK.get(str(ticker).strip(), len(_TICKER_RANK) + 1)

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

def call_llm(prompt, temperature=None):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    if temperature is not None:
        body["temperature"] = float(temperature)

    while True:
        try:
            r = requests.post(OLLAMA_API_URL, json=body, timeout=600)

            if r.status_code == 200:
                try:
                    return r.json()["choices"][0]["message"]["content"]
                except:
                    return None

            elif r.status_code == 429:
                wait = 4 + random.random() * 2
                print(f"⚠️ Rate limited, waiting {wait:.1f}s — {r.text}")
                time.sleep(wait)
                continue

            else:
                print("❌ API error:", r.status_code, r.text)
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
    text = text.strip()[:3000]

    return f"""You are an expert financial analyst. Your task is to extract specific causal factors from the article below that could affect the stock price of {ticker}.

{company_context}

Temporal restriction: Pretend today is {date}. Use ONLY information that would be known on or before that date. NO hindsight.

Article:
\"\"\"{text}\"\"\"

Instructions:
- "relevance": a float from 0.0 to 1.0 indicating how directly this article relates to {ticker}
  - 0.8–1.0: article is primarily about {ticker}, discusses its earnings/products/strategy directly
  - 0.5–0.7: article discusses {ticker} substantially alongside other companies
  - 0.2–0.4: article mentions {ticker} but focuses on its sector, competitors, or a related topic
  - 0.0–0.1: {ticker} is mentioned in passing (in a list, disclaimer, or brief comparison)
- "relevance_reasoning": one sentence explaining why you gave this relevance score
- Identify 3 to 5 specific factors from this article that could influence {ticker}'s stock price.
- For each factor, provide:
  - "factor": a short description of the causal factor (1-2 sentences)
  - "direction": "positive" or "negative" (the expected impact on {ticker}'s stock price)
  - "time_horizon": "short-term" (1-5 days), "medium-term" (1-4 weeks), or "long-term" (months+)
  - "confidence": "high", "medium", or "low"
- If the article is NOT directly about {ticker}, still identify indirect effects (industry trends, competitor news, macro factors) but mark confidence as "low" or "medium".
- Consider supply chain effects, competitive dynamics, regulatory implications, and market sentiment.

You MUST output ONLY a valid JSON object. NO markdown. NO code fences. NO commentary.

Return this structure:
{{
  "relevance": 0.0 to 1.0,
  "relevance_reasoning": "one sentence",
  "factors": [
    {{"factor": "...", "direction": "positive", "time_horizon": "short-term", "confidence": "high"}},
    {{"factor": "...", "direction": "negative", "time_horizon": "medium-term", "confidence": "medium"}}
  ]
}}"""


def format_factors_for_prediction(factors_data):
    """Format extracted factors into readable text for the prediction prompt."""
    if isinstance(factors_data, dict):
        factors_list = factors_data.get("factors", [])
    elif isinstance(factors_data, list):
        factors_list = factors_data
    else:
        return "Factor extraction failed. Analyze the article directly."

    if not factors_list:
        return "No specific factors identified. Analyze the article directly."

    lines = ["Extracted causal factors:"]
    for i, f in enumerate(factors_list, 1):
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
                            price_summary, company_context, factors_text,
                            relevance, relevance_reasoning):
    """Build prompt for the second LLM call: predict direction using extracted factors.

    The prompt integrates company context, pre-computed price indicators, and
    the causal factors from step 1, with horizon-specific reasoning guidance
    (FinGPT, Liang et al., 2024).
    """
    text = text.strip()[:3000]
    horizon_instruction = HORIZON_INSTRUCTIONS.get(horizon, "")

    return f"""You are an expert financial analyst predicting stock price direction for {ticker}.

{company_context}

Article relevance to {ticker}: {relevance:.2f} — {relevance_reasoning}

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
  "confidence": number between 0.0 and 1.0,
  "explanation": "short explanation"
}}

Rules:
- direction MUST be either "UP" or "DOWN"
- Even if the news seems mixed or loosely related, commit to whichever direction is more likely
- confidence: a number from 0.0 to 1.0 representing how confident you are in the DIRECTION prediction
  - 0.8–1.0: Strong conviction — clear directional signal
  - 0.6–0.8: Moderate conviction — likely direction but some uncertainty
  - 0.4–0.6: Low conviction — mixed signals, could go either way
  - Below 0.4: Very low conviction — essentially guessing
- confidence is about DIRECTION certainty, NOT about price magnitude
- explanation: 1-2 sentences justifying the direction
- ALWAYS include all 4 fields"""


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

    Returns a dict:
        {"predictions": {horizon: pred_dict_or_None},
         "relevance": float or None,
         "relevance_reasoning": str or None}
    Relevance is per-row (horizon-independent) and is now written to a top-level
    column rather than embedded inside each pred_{horizon} JSON.
    """
    result = {"predictions": {}, "relevance": None, "relevance_reasoning": None}

    article_text = clean_article_text(row.get("Article_text", ""))
    if len(article_text) < 150:
        return result

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

    factors_data = None
    relevance = 0.0
    relevance_reasoning = ""

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(THROTTLE_SEC)
        raw = call_llm(factor_prompt)
        clean = clean_raw_output(raw)
        parsed = extract_json(clean)
        if isinstance(parsed, dict) and "factors" in parsed:
            factors_data = parsed
            relevance = float(parsed.get("relevance", 0.0))
            relevance = max(0.0, min(1.0, relevance))
            relevance_reasoning = str(parsed.get("relevance_reasoning", ""))
            break
        elif isinstance(parsed, list):
            # Backward compat: old format was just a list of factors
            factors_data = {"relevance": 0.5, "relevance_reasoning": "unknown", "factors": parsed}
            relevance = 0.5
            relevance_reasoning = "unknown"
            break
        print(f"      ⚠️ Factor extraction parse failed (attempt {attempt})")

    if factors_data is None:
        print(f"    ⚠️ Factor extraction failed after {MAX_RETRIES} attempts")
        factors_data = {"relevance": 0.0, "relevance_reasoning": "extraction failed", "factors": []}
        relevance = 0.0
        relevance_reasoning = "extraction failed"

    result["relevance"] = relevance
    result["relevance_reasoning"] = relevance_reasoning

    # Relevance gating — skip all 5 prediction calls if article isn't relevant enough
    if relevance < RELEVANCE_THRESHOLD:
        print(f"    ⚠️ Low relevance ({relevance:.2f}): {relevance_reasoning} — skipping predictions")
        skip_result = {
            "article_id": article_id,
            "ticker": ticker,
            "direction": "SKIPPED",
            "confidence": 0.0,
            "explanation": f"Article relevance ({relevance:.2f}) below threshold ({RELEVANCE_THRESHOLD})",
        }
        for horizon in HORIZONS:
            result["predictions"][horizon] = skip_result
        return result

    print(f"    Relevance: {relevance:.2f} — proceeding with predictions")
    factors_text = format_factors_for_prediction(factors_data)

    # 4. Per-horizon predictions (with optional self-consistency sampling)
    for horizon in HORIZONS:
        print(f"    {horizon}...")

        prompt = build_prediction_prompt(
            article_id, date, ticker, article_text, horizon,
            price_summary, company_context, factors_text,
            relevance, relevance_reasoning,
        )

        # Collect K_SAMPLES samples (each with its own retry budget)
        samples = []
        sample_temp = SAMPLE_TEMPERATURE if K_SAMPLES > 1 else None
        for k in range(K_SAMPLES):
            sample = None
            for attempt in range(1, MAX_RETRIES + 1):
                time.sleep(THROTTLE_SEC)
                raw = call_llm(prompt, temperature=sample_temp)
                clean = clean_raw_output(raw)
                parsed = extract_json(clean)
                if isinstance(parsed, dict) and parsed.get("direction") in ("UP", "DOWN"):
                    sample = parsed
                    break
                print(f"      ⚠️ Parse failed (sample {k+1}/{K_SAMPLES}, attempt {attempt})")
            if sample is not None:
                samples.append(sample)

        if not samples:
            result["predictions"][horizon] = None
            continue

        if K_SAMPLES == 1:
            # Legacy: keep the model's self-rated confidence
            final = samples[0]
        else:
            # Majority vote with vote-agreement as calibrated confidence
            up_count = sum(1 for s in samples if s.get("direction") == "UP")
            down_count = sum(1 for s in samples if s.get("direction") == "DOWN")
            winning_dir = "UP" if up_count >= down_count else "DOWN"
            agreement = max(up_count, down_count) / len(samples)
            winning = next(s for s in samples if s.get("direction") == winning_dir)
            final = {
                **winning,
                "direction": winning_dir,
                "confidence": float(agreement),
                "n_samples": len(samples),
                "vote_up": up_count,
                "vote_down": down_count,
            }

        # NOTE: relevance no longer copied into the per-horizon JSON — it's a top-level column.
        # Strip the keys in case the LLM echoed them in its output.
        final.pop("relevance", None)
        final.pop("relevance_reasoning", None)
        result["predictions"][horizon] = final

    return result


# ============================================================
# FILE PROCESSING & MAIN
# ============================================================

def _row_skipped_relevance(row):
    """If a row was previously SKIPPED across all horizons, return the stored
    relevance value (float). Returns None if the row was not fully SKIPPED
    or if no stored relevance can be parsed.
    """
    # 1. Confirm every horizon is SKIPPED.
    for h in HORIZONS:
        pred_json = row.get(f"pred_{h}")
        if not pred_json or not isinstance(pred_json, str) or pred_json.strip() == "":
            return None
        try:
            pred = json.loads(pred_json)
        except Exception:
            return None
        if pred.get("direction") != "SKIPPED":
            return None

    # 2. Read relevance — prefer the dedicated column, fall back to JSON for
    #    parquets that haven't been backfilled yet.
    rel = row.get("relevance")
    if pd.notna(rel):
        try:
            return float(rel)
        except (TypeError, ValueError):
            pass

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
            return float(rel)
        except (TypeError, ValueError):
            return None

    return None


def _load_or_init_pred_df(return_path):
    """Load the prediction parquet (resume) or the source parquet (fresh)."""
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if os.path.exists(pred_path):
        df = pd.read_parquet(pred_path)
        is_resume = True
    else:
        df = pd.read_parquet(return_path)
        is_resume = False

    for horizon in HORIZONS:
        if f"pred_{horizon}" not in df.columns:
            df[f"pred_{horizon}"] = None
    # Top-level relevance columns (per-row, horizon-independent)
    if "relevance" not in df.columns:
        df["relevance"] = None
    if "relevance_reasoning" not in df.columns:
        df["relevance_reasoning"] = None
    return df, pred_path, is_resume


def predict_file(return_path, target_ticker=None):
    """Run predictions for one batch parquet.

    If target_ticker is given, only rows whose Stock_symbol matches are
    predicted. Otherwise all remaining rows are predicted.
    The eval CSV is written by write_eval_csv() in a separate pass.
    """
    df, pred_path, is_resume = _load_or_init_pred_df(return_path)

    # Re-queue previously SKIPPED rows whose stored relevance now meets the
    # current RELEVANCE_THRESHOLD (e.g. user lowered it from 0.4 to 0.2).
    requeued = 0
    for idx in range(len(df)):
        stored_rel = _row_skipped_relevance(df.iloc[idx])
        if stored_rel is not None and stored_rel >= RELEVANCE_THRESHOLD:
            for h in HORIZONS:
                df.at[idx, f"pred_{h}"] = None
            requeued += 1

    # Determine which row indices we'll actually process
    if target_ticker is not None:
        tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
        candidate_indices = [i for i, t in enumerate(tickers_in_df) if t == target_ticker]
        if not candidate_indices and requeued == 0:
            return  # nothing to do in this file
    else:
        candidate_indices = list(range(len(df)))

    # Quick scan: any rows actually need work?
    todo = []
    for idx in candidate_indices:
        row = df.iloc[idx]
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue
        all_done = all(
            row.get(f"pred_{h}") is not None
            and isinstance(row.get(f"pred_{h}"), str)
            and row.get(f"pred_{h}").strip() != ""
            for h in HORIZONS
        )
        if not all_done:
            todo.append(idx)

    if not todo and requeued == 0:
        return  # nothing to do, don't even print

    label = f"target={target_ticker}" if target_ticker else "all tickers"
    print(f"📈 {os.path.basename(return_path)}  ({label}) — {len(todo)} rows to predict")
    if is_resume:
        print(f"  ♻️ Resuming from checkpoint")
    if requeued > 0:
        print(f"  🔁 Re-queued {requeued} previously SKIPPED rows "
              f"(relevance ≥ current threshold {RELEVANCE_THRESHOLD})")

    processed = 0
    for idx in todo:
        row = df.iloc[idx]
        ticker = str(row.get("Stock_symbol", "")).strip()
        date = str(row.get("Date", ""))[:10]

        if USE_ENHANCED_PIPELINE:
            result = predict_for_row(row)
            predictions = result["predictions"]
            # Write per-row relevance metadata to the dedicated columns
            if result["relevance"] is not None:
                df.at[idx, "relevance"] = float(result["relevance"])
            if result["relevance_reasoning"] is not None:
                df.at[idx, "relevance_reasoning"] = result["relevance_reasoning"]
        else:
            hist_text = load_historical_prices_legacy(ticker, date)
            predictions = predict_for_row_legacy(row, hist_text)
            # Legacy pipeline doesn't compute relevance — leave the columns alone.

        for horizon, pred in predictions.items():
            if pred is not None:
                df.at[idx, f"pred_{horizon}"] = json.dumps(pred)

        processed += 1
        print(f"  ✅ {processed}/{len(todo)} done")

        # Checkpoint after each row
        pq.write_table(pa.Table.from_pandas(df), pred_path)

    # Final save
    pq.write_table(pa.Table.from_pandas(df), pred_path)


def write_eval_csv(return_path):
    """Read the current prediction parquet and write the evaluation CSV."""
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if not os.path.exists(pred_path):
        return
    df = pd.read_parquet(pred_path)

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

        for horizon in HORIZONS:
            pred_json = row.get(f"pred_{horizon}")
            if pred_json and isinstance(pred_json, str):
                try:
                    pred = json.loads(pred_json)
                    direction = pred.get("direction")
                    eval_row[f"Pred_Direction_{horizon}"] = direction
                    if direction == "SKIPPED":
                        eval_row[f"Pred_Confidence_{horizon}"] = 0.0
                    else:
                        eval_row[f"Pred_Confidence_{horizon}"] = pred.get("confidence")
                    eval_row[f"Explanation_{horizon}"] = pred.get("explanation")
                except:
                    eval_row[f"Pred_Direction_{horizon}"] = None
                    eval_row[f"Pred_Confidence_{horizon}"] = None
                    eval_row[f"Explanation_{horizon}"] = None
            else:
                eval_row[f"Pred_Direction_{horizon}"] = None
                eval_row[f"Pred_Confidence_{horizon}"] = None
                eval_row[f"Explanation_{horizon}"] = None

        # Prefer the top-level relevance column (canonical source going forward).
        # Fall back to the per-horizon JSON only when the column hasn't been
        # backfilled on this parquet yet.
        rel_col = row.get("relevance")
        reason_col = row.get("relevance_reasoning")
        if pd.notna(rel_col):
            relevance_val = float(rel_col)
            relevance_reason = reason_col if pd.notna(reason_col) else None
        else:
            relevance_val = None
            relevance_reason = None
            for horizon in HORIZONS:
                pred_json = row.get(f"pred_{horizon}")
                if pred_json and isinstance(pred_json, str):
                    try:
                        pred = json.loads(pred_json)
                        if relevance_val is None:
                            relevance_val = pred.get("relevance")
                            relevance_reason = pred.get("relevance_reasoning")
                    except:
                        pass
        eval_row["Pred_Relevance"] = relevance_val
        eval_row["Relevance_Reasoning"] = relevance_reason

        for horizon in HORIZONS:
            eval_row[f"Actual_Return_{horizon}"] = row.get(f"return_{horizon}")
            eval_row[f"Actual_Direction_{horizon}"] = row.get(f"actual_direction_{horizon}")
            eval_row[f"Actual_Strength_{horizon}"] = row.get(f"actual_strength_{horizon}")
            eval_row[f"Actual_Majority_Direction_{horizon}"] = row.get(f"actual_majority_direction_{horizon}")
            eval_row[f"Actual_Up_Days_{horizon}"] = row.get(f"actual_up_days_{horizon}")
            eval_row[f"Actual_Down_Days_{horizon}"] = row.get(f"actual_down_days_{horizon}")

        eval_rows.append(eval_row)

    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(EVAL_DIR, os.path.basename(return_path).replace(".parquet", "_eval.csv"))
    eval_df.to_csv(eval_path, index=False)
    print(f"  📄 Wrote eval CSV: {eval_path}")


def index_files_by_ticker(files):
    """Scan all batch parquets once to build {ticker: [file, ...]}.

    Lets us skip files that don't contain the current target ticker, which
    avoids re-reading every parquet 69 times.
    """
    print("🔎 Indexing batches by ticker...")
    ticker_files = {}
    for f in files:
        path = os.path.join(RETURN_DIR, f)
        try:
            df = pd.read_parquet(path, columns=["Stock_symbol"])
        except Exception as e:
            print(f"  ⚠️ Could not read {f}: {e}")
            continue
        for t in df["Stock_symbol"].dropna().astype(str).str.strip().unique():
            ticker_files.setdefault(t, []).append(f)
    print(f"  Indexed {sum(len(v) for v in ticker_files.values())} (ticker, file) pairs across {len(ticker_files)} tickers")
    return ticker_files


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No return batch parquet files found.")
        return

    pipeline_label = "enhanced" if USE_ENHANCED_PIPELINE else "legacy"
    print(f"Pipeline: {pipeline_label}")

    ticker_files = index_files_by_ticker(files)

    # Order all observed tickers by priority (priority list first, anything
    # else after, alphabetized so unranked tickers have a stable order).
    observed = set(ticker_files.keys())
    ordered_priority = [t for t in TICKER_PRIORITY if t in observed]
    leftover = sorted(observed - set(TICKER_PRIORITY))
    ticker_order = ordered_priority + leftover

    print(f"\nTicker processing order ({len(ticker_order)} total):")
    for i, t in enumerate(ticker_order, 1):
        marker = "" if t in _TICKER_RANK else "  (unranked)"
        print(f"  {i:>3}. {t}{marker}  — {len(ticker_files[t])} batches")

    # Phase 1: predict ticker-by-ticker, globally across all batches.
    for i, ticker in enumerate(ticker_order, 1):
        print(f"\n🎯 [{i}/{len(ticker_order)}] Predicting ticker: {ticker}")
        for f in ticker_files[ticker]:
            return_path = os.path.join(RETURN_DIR, f)
            predict_file(return_path, target_ticker=ticker)

    # Phase 2: write eval CSVs once at the end.
    print("\n📄 Writing evaluation CSVs...")
    for f in files:
        write_eval_csv(os.path.join(RETURN_DIR, f))


if __name__ == "__main__":
    main()
