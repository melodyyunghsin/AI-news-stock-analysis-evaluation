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
MODEL   = "models/gemini-2.5-flash-lite"
_call_count = 0

RETURN_DIR = "data/balanced_focused_dataset"  # Use pre-cleaned articles
PRED_DIR   = "data/balanced_focused_predictions_gemini_k5"
EVAL_DIR   = "data/balanced_focused_evaluation_results_gemini_k5"
PRICE_DIR  = "data/full_history"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

THROTTLE_SEC = 3  # Stay within Gemini free tier rate limits
MAX_RETRIES  = 3

HORIZONS = ["1d", "3d", "5d", "10d", "21d"]

RELEVANCE_THRESHOLD = 0.0  # Drop candidates below this before balanced selection

# Per-ticker cap for balanced selection. We pick up to MAX_ARTICLES_PER_TICKER // 2
# UP articles and the same number of DOWN articles, ranked by relevance
# descending. See PER_HORIZON_BALANCED_SELECTION for the direction criterion.
MAX_ARTICLES_PER_TICKER = 100

# How rows are classified UP/DOWN for the balanced top-N pick:
#   True  → per-horizon: run select_balanced_top_n once per horizon (each picks
#           50 UP-at-h + 50 DOWN-at-h) and union the 5 selections. Each
#           horizon's eval set ends up exactly 50/50 at that horizon.
#   False → 5-horizon majority: classify each row by the majority of UP vs
#           DOWN across all 5 horizons (legacy). Per-horizon eval splits can
#           be lopsided (e.g. TSLA 1d ended up 82/18 actual UP/DOWN).
PER_HORIZON_BALANCED_SELECTION = True

# When True, selection ALSO balances on article tone (POS = more positive
# factors than negative, NEG = vice versa). 4-way quadrant split:
# (POS×UP, POS×DOWN, NEG×UP, NEG×DOWN), each capped at MAX_ARTICLES_PER_TICKER//4.
# Eliminates the ~78% positive-toned input bias at the cost of ~20-40% smaller
# eval set per horizon. Only effective when PER_HORIZON_BALANCED_SELECTION=True.
FOUR_WAY_TONE_BALANCED_SELECTION = True

# Self-consistency sampling for the prediction step.
# K_SAMPLES = 1: legacy behavior (one call per horizon, model self-rates confidence).
# K_SAMPLES > 1: call k times at SAMPLE_TEMPERATURE, majority-vote the direction,
#                use vote agreement (e.g. 4/5 = 0.8) as the calibrated confidence.
# K=5 typically adds 3-5pp accuracy and produces reliable calibration, at 5x cost.
K_SAMPLES = 11  # was 5; bumped to reduce run-to-run vote-flipping variance on borderline rows
SAMPLE_TEMPERATURE = 0.8

# Toggle between enhanced two-step pipeline and legacy single-prompt pipeline.
# Note: the balanced top-N selection runs only in the enhanced pipeline (it
# depends on the relevance score that comes out of factor extraction).
USE_ENHANCED_PIPELINE = True

# When True, Phase 1 (factor extraction) and Phase 3 (per-horizon prediction)
# are routed through the Gemini Batch API instead of synchronous per-row calls.
# Batch is ~50% cheaper and async; turnaround is minutes-to-hours instead of
# seconds. Phase 2 (selection) is local and unaffected. The legacy pipeline
# (USE_ENHANCED_PIPELINE=False) always uses the sync path.
# Endpoint reference: https://ai.google.dev/gemini-api/docs/batch-api
USE_BATCH_API = True
BATCH_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
BATCH_POLL_INTERVAL_SEC = 30  # how often to poll the batch op for completion

# Tickers are predicted in this order across ALL batches. If you stop the run
# partway through, the highest-priority tickers will be fully covered globally
# rather than partially covered across every ticker.
TICKER_PRIORITY = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOG", "TSM"]
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

def get_api_url():
    global _call_count
    key = API_KEYS[_call_count % len(API_KEYS)]
    _call_count += 1
    return f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={key}"

def call_llm(prompt, temperature=None):
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    if temperature is not None:
        body["generationConfig"] = {"temperature": float(temperature)}

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
                print("❌ API error:", r.status_code, r.text)
                time.sleep(2)

        except Exception as e:
            print("❌ Network error:", e)
            time.sleep(3)


# ============================================================
# BATCH API HELPERS
# ============================================================
# The Gemini Batch API runs a list of generateContent requests asynchronously
# at ~50% the synchronous price. We build a list of {key, request} entries,
# submit them in one POST, then poll the returned operation until SUCCEEDED.
# Results come back ordered identically to the submitted requests; we use the
# client-side `key` field (e.g. "factor|<file>|<idx>" or
# "pred|<file>|<idx>|<horizon>|k<sample>") to correlate them back to the row
# they originated from.
#
# IMPORTANT: the exact JSON shape of the batch API has evolved. The structure
# below matches the v1beta inline-requests shape documented at
# https://ai.google.dev/gemini-api/docs/batch-api — verify before running.
# `_extract_inlined_responses` reads two common response layouts so minor
# server-side renames don't break the pipeline.


def _next_api_key():
    """Round-robin one of the configured API keys for the next batch op.
    Each submit/poll consumes one tick of the same counter used by sync calls.
    """
    global _call_count
    key = API_KEYS[_call_count % len(API_KEYS)]
    _call_count += 1
    return key


def build_batch_request(key, prompt, temperature=None):
    """Build one entry for a batch submission.

    Returns: {"key": <client identifier>, "request": <generateContent body>}
    The `key` is purely client-side bookkeeping so we can map the i-th
    response back to (row, horizon, sample) after the job completes.
    """
    request_body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
    }
    if temperature is not None:
        request_body["generationConfig"] = {"temperature": float(temperature)}
    return {"key": key, "request": request_body}


def submit_batch(batch_requests, display_name="gemini-batch"):
    """POST a batch of {key, request} entries. Returns (operation_name, ordered_keys).

    ordered_keys preserves submission order so we can correlate the i-th
    inlined response back to its client key. Retries on 429 with backoff.
    """
    if not batch_requests:
        return None, []

    api_key = _next_api_key()
    url = f"{BATCH_API_BASE}/{MODEL}:batchGenerateContent?key={api_key}"

    ordered_keys = [r["key"] for r in batch_requests]
    body = {
        "batch": {
            "displayName": f"{display_name}-{int(time.time())}",
            "inputConfig": {
                "requests": {
                    "requests": [{"request": r["request"]} for r in batch_requests]
                }
            },
        }
    }

    while True:
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                op_name = r.json().get("name")
                print(f"  📤 Batch submitted: {display_name} "
                      f"({len(ordered_keys)} reqs) → {op_name}")
                return op_name, ordered_keys
            elif r.status_code == 429:
                wait = 10 + random.random() * 5
                print(f"  ⚠️ Submit rate-limited, waiting {wait:.1f}s — {r.text}")
                time.sleep(wait)
                continue
            else:
                print(f"  ❌ Batch submit error {r.status_code}: {r.text}")
                time.sleep(5)
        except Exception as e:
            print(f"  ❌ Batch submit network error: {e}")
            time.sleep(5)


def _extract_inlined_responses(poll_data):
    """Pull the per-request response list out of a SUCCEEDED batch poll body.

    Handles both `response.inlinedResponses.inlinedResponses` (older shape)
    and `response.responses` / `response.inlinedResponses` (newer shapes).
    Returns a list; each entry is a dict with either "response" or "error".
    """
    resp = poll_data.get("response", {})
    candidates = (
        resp.get("inlinedResponses", {}).get("inlinedResponses")
        or resp.get("inlinedResponses")
        or resp.get("responses")
        or []
    )
    return candidates if isinstance(candidates, list) else []


def poll_batch(operation_name, ordered_keys, poll_interval=None):
    """Poll a batch operation until SUCCEEDED / FAILED / CANCELLED.

    Returns {client_key: response_text_or_None}. Per-request errors and missing
    entries map to None so the caller can treat them as parse failures.
    """
    if operation_name is None or not ordered_keys:
        return {k: None for k in ordered_keys}
    if poll_interval is None:
        poll_interval = BATCH_POLL_INTERVAL_SEC

    api_key = _next_api_key()
    url = f"{BATCH_API_BASE}/{operation_name}?key={api_key}"

    elapsed = 0
    while True:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  ⚠️ Poll error {r.status_code}: {r.text}")
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue

            data = r.json()
            state = (
                data.get("metadata", {}).get("state")
                or data.get("state")
                or ""
            )
            done = bool(data.get("done")) or state.endswith("SUCCEEDED")

            if done and not (state.endswith("FAILED") or state.endswith("CANCELLED")):
                inlined = _extract_inlined_responses(data)
                results = {}
                for i, key in enumerate(ordered_keys):
                    if i >= len(inlined):
                        results[key] = None
                        continue
                    entry = inlined[i]
                    if "error" in entry:
                        results[key] = None
                        continue
                    resp = entry.get("response", entry)
                    try:
                        results[key] = resp["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError, TypeError):
                        results[key] = None
                ok = sum(1 for v in results.values() if v)
                print(f"  ✅ Batch {operation_name} done — {ok}/{len(ordered_keys)} OK")
                return results

            if state.endswith("FAILED") or state.endswith("CANCELLED"):
                print(f"  ❌ Batch {operation_name} ended in state {state}")
                return {k: None for k in ordered_keys}

            print(f"  ⏳ Batch {operation_name} state={state or '?'} "
                  f"(elapsed {elapsed}s, polling every {poll_interval}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval

        except Exception as e:
            print(f"  ⚠️ Poll network error: {e}")
            time.sleep(poll_interval)
            elapsed += poll_interval


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
        "© 2025, Nasdaq, Inc.",
        "© 2024, Nasdaq, Inc.",
        "© 2023, Nasdaq, Inc.",
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
# TWO-STEP PREDICTION — split into factor extraction + per-horizon prediction
# so we can do balanced top-N selection in between.
# ============================================================

def extract_factors_for_row(row):
    """Step 1: factor extraction only.

    Returns (factors_data_dict, relevance, relevance_reasoning) on success,
    or None if extraction failed or the article is too short.
    """
    article_text = clean_article_text(row.get("Article_text", ""))
    if len(article_text) < 150:
        return None

    ticker = str(row.get("Stock_symbol", "")).strip()
    date = str(row.get("Date", ""))[:10]

    company_context = get_company_context(ticker)
    factor_prompt = build_factor_extraction_prompt(ticker, article_text, company_context, date)

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(THROTTLE_SEC)
        raw = call_llm(factor_prompt)
        clean = clean_raw_output(raw)
        parsed = extract_json(clean)
        if isinstance(parsed, dict) and "factors" in parsed:
            relevance = float(parsed.get("relevance", 0.0))
            relevance = max(0.0, min(1.0, relevance))
            relevance_reasoning = str(parsed.get("relevance_reasoning", ""))
            return parsed, relevance, relevance_reasoning
        elif isinstance(parsed, list):
            # Backward compat: old format was just a list of factors
            return ({"relevance": 0.5, "relevance_reasoning": "unknown", "factors": parsed},
                    0.5, "unknown")
        print(f"      ⚠️ Factor extraction parse failed (attempt {attempt})")

    return None


def predict_horizons_for_row(article_id, ticker, date, article_text,
                             factors_data, relevance, relevance_reasoning):
    """Step 2: run per-horizon prediction (with self-consistency sampling).

    Returns dict mapping horizon → prediction dict (or None on parse failure).
    """
    company_context = get_company_context(ticker)
    price_summary = build_price_summary(ticker, date)
    factors_text = format_factors_for_prediction(factors_data)

    predictions = {}
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
            predictions[horizon] = None
            continue

        if K_SAMPLES == 1:
            final = samples[0]
        else:
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

        # Strip relevance keys if the LLM echoed them — they live on the row.
        final.pop("relevance", None)
        final.pop("relevance_reasoning", None)
        predictions[horizon] = final

    return predictions


# ============================================================
# BATCH-API VARIANTS of factor extraction and per-horizon prediction
# ============================================================
# These mirror the sync `extract_factors_for_row` + `predict_horizons_for_row`
# but submit one big batch covering ALL outstanding work for a single ticker
# (across every batch parquet that contains the ticker) and apply results in a
# single bulk write per file when the job returns. Parse/retry logic still
# lives in extract_json + clean_raw_output; a per-key parse failure leaves the
# corresponding pred or relevance field as None just like the sync path does.


def extract_factors_batch(ticker, files_for_ticker):
    """Phase 1 via Batch API. Submits one batch for every row of `ticker`
    across all files that doesn't yet have stored factors/relevance.
    """
    file_dfs = {}        # return_path -> df
    file_paths = {}      # return_path -> pred_path
    plan = []            # list of (return_path, idx, prompt)

    for f in files_for_ticker:
        return_path = os.path.join(RETURN_DIR, f)
        df, pred_path, _ = _load_or_init_pred_df(return_path)
        file_dfs[return_path] = df
        file_paths[return_path] = pred_path

        tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
        for idx, t in enumerate(tickers_in_df):
            if t != ticker:
                continue
            row = df.iloc[idx]
            article_text = row.get("Article_text", "")
            if not isinstance(article_text, str) or article_text.strip() == "":
                continue
            if _has_factor_extraction(row):
                continue
            cleaned = clean_article_text(article_text)
            if len(cleaned) < 150:
                continue
            date = str(row.get("Date", ""))[:10]
            company_context = get_company_context(ticker)
            prompt = build_factor_extraction_prompt(ticker, cleaned, company_context, date)
            plan.append((return_path, idx, prompt))

    if not plan:
        # Persist any newly-added columns and exit.
        for return_path, df in file_dfs.items():
            pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
        return

    print(f"  📥 Batch factor-extraction for {ticker}: {len(plan)} rows")

    batch_requests = [
        build_batch_request(
            key=f"factor|{os.path.basename(rp)}|{idx}",
            prompt=prompt,
        )
        for (rp, idx, prompt) in plan
    ]
    op_name, ordered_keys = submit_batch(batch_requests, display_name=f"factors-{ticker}")
    results = poll_batch(op_name, ordered_keys)

    for (rp, idx, _prompt) in plan:
        key = f"factor|{os.path.basename(rp)}|{idx}"
        raw = results.get(key)
        clean = clean_raw_output(raw) if raw else None
        parsed = extract_json(clean) if clean else None

        df = file_dfs[rp]
        if isinstance(parsed, dict) and "factors" in parsed:
            relevance = float(parsed.get("relevance", 0.0))
            relevance = max(0.0, min(1.0, relevance))
            reasoning = str(parsed.get("relevance_reasoning", ""))
            df.at[idx, "relevance"] = relevance
            df.at[idx, "relevance_reasoning"] = reasoning
            df.at[idx, "factors_json"] = json.dumps(parsed)
        elif isinstance(parsed, list):
            # Backward compat: old-format response was just a list of factors.
            wrapped = {"relevance": 0.5, "relevance_reasoning": "unknown", "factors": parsed}
            df.at[idx, "relevance"] = 0.5
            df.at[idx, "relevance_reasoning"] = "unknown"
            df.at[idx, "factors_json"] = json.dumps(wrapped)
        else:
            df.at[idx, "relevance"] = 0.0
            df.at[idx, "relevance_reasoning"] = "extraction failed"
            df.at[idx, "factors_json"] = json.dumps({"factors": []})

    # One bulk write per file when the whole batch is done.
    for return_path, df in file_dfs.items():
        pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
    print(f"  💾 Wrote factor results for {ticker} to {len(file_dfs)} parquet(s)")


def predict_horizons_batch(ticker, files_for_ticker, selected_keys):
    """Phase 3 via Batch API. Submits one batch covering
    (selected rows) × HORIZONS × K_SAMPLES, then majority-votes per (row, horizon).
    """
    file_dfs = {}
    file_paths = {}
    # Each entry: (return_path, idx, horizon, prompt). One per (row, horizon)
    # that still needs prediction; K_SAMPLES copies are submitted per entry.
    plan = []

    for f in files_for_ticker:
        return_path = os.path.join(RETURN_DIR, f)
        df, pred_path, _ = _load_or_init_pred_df(return_path)
        file_dfs[return_path] = df
        file_paths[return_path] = pred_path

        tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
        ticker_indices = [i for i, t in enumerate(tickers_in_df) if t == ticker]
        if not ticker_indices:
            continue

        for idx in ticker_indices:
            is_selected = (return_path, idx) in selected_keys
            df.at[idx, "selected"] = bool(is_selected)
            # Clear stale SKIPPED markers from older runs so selected rows
            # get re-predicted with the new pipeline.
            for h in HORIZONS:
                pred_json = df.at[idx, f"pred_{h}"]
                if isinstance(pred_json, str) and pred_json.strip():
                    try:
                        old = json.loads(pred_json)
                    except Exception:
                        old = None
                    if isinstance(old, dict) and old.get("direction") == "SKIPPED":
                        df.at[idx, f"pred_{h}"] = None
            if not is_selected:
                continue

            row = df.iloc[idx]
            article_id = str(row.get("Article_id", ""))
            date = str(row.get("Date", ""))[:10]
            article_text = clean_article_text(row.get("Article_text", ""))
            relevance = float(row.get("relevance", 0.0))
            relevance_reasoning = str(row.get("relevance_reasoning") or "")

            try:
                factors_data = json.loads(row.get("factors_json") or "{}")
            except Exception:
                factors_data = {"factors": []}

            company_context = get_company_context(ticker)
            price_summary = build_price_summary(ticker, date)
            factors_text = format_factors_for_prediction(factors_data)

            for horizon in HORIZONS:
                pred_json = df.at[idx, f"pred_{horizon}"]
                if isinstance(pred_json, str) and pred_json.strip():
                    continue  # already done — skip
                prompt = build_prediction_prompt(
                    article_id, date, ticker, article_text, horizon,
                    price_summary, company_context, factors_text,
                    relevance, relevance_reasoning,
                )
                plan.append((return_path, idx, horizon, prompt))

    if not plan:
        for return_path, df in file_dfs.items():
            pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
        return

    sample_temp = SAMPLE_TEMPERATURE if K_SAMPLES > 1 else None
    batch_requests = []
    for (rp, idx, horizon, prompt) in plan:
        for k in range(K_SAMPLES):
            key = f"pred|{os.path.basename(rp)}|{idx}|{horizon}|k{k}"
            batch_requests.append(
                build_batch_request(key, prompt, temperature=sample_temp)
            )

    print(f"  📤 Batch predictions for {ticker}: {len(plan)} (row,horizon) "
          f"× K={K_SAMPLES} = {len(batch_requests)} requests")

    op_name, ordered_keys = submit_batch(batch_requests, display_name=f"preds-{ticker}")
    results = poll_batch(op_name, ordered_keys)

    # Aggregate K_SAMPLES per (row, horizon): majority vote, vote-agreement
    # becomes calibrated confidence (matches sync `predict_horizons_for_row`).
    for (rp, idx, horizon, _prompt) in plan:
        samples = []
        for k in range(K_SAMPLES):
            key = f"pred|{os.path.basename(rp)}|{idx}|{horizon}|k{k}"
            raw = results.get(key)
            clean = clean_raw_output(raw) if raw else None
            parsed = extract_json(clean) if clean else None
            if isinstance(parsed, dict) and parsed.get("direction") in ("UP", "DOWN"):
                samples.append(parsed)

        df = file_dfs[rp]
        if not samples:
            # Leave pred_{horizon} as None (matches sync behavior on failure).
            continue

        if K_SAMPLES == 1:
            final = samples[0]
        else:
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
        final.pop("relevance", None)
        final.pop("relevance_reasoning", None)
        df.at[idx, f"pred_{horizon}"] = json.dumps(final)

    for return_path, df in file_dfs.items():
        pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
    print(f"  💾 Wrote prediction results for {ticker} to {len(file_dfs)} parquet(s)")


# ============================================================
# BALANCED TOP-N SELECTION
# ============================================================

def majority_actual_direction(row):
    """Classify a row's "actual majority direction" across all 5 horizons.

    UP if strictly more horizons closed UP than DOWN, DOWN if strictly more
    closed DOWN, None on ties or when no horizon has a direction recorded.
    With 5 horizons and complete data ties are impossible; ties only occur
    when one or more actual_direction_* columns are null.
    """
    ups = sum(1 for h in HORIZONS if row.get(f"actual_direction_{h}") == "UP")
    downs = sum(1 for h in HORIZONS if row.get(f"actual_direction_{h}") == "DOWN")
    if ups == 0 and downs == 0:
        return None
    if ups > downs:
        return "UP"
    if downs > ups:
        return "DOWN"
    return None  # tie — discard


def per_horizon_actual_direction(row, horizon):
    """Read actual_direction_{horizon} → 'UP'/'DOWN'/None.
    Used by per-horizon balanced selection.
    """
    d = row.get(f"actual_direction_{horizon}")
    if d == "UP":
        return "UP"
    if d == "DOWN":
        return "DOWN"
    return None


def select_balanced_top_n(candidates, max_articles=MAX_ARTICLES_PER_TICKER):
    """Given (return_path, idx, relevance, majority_dir) tuples, return the set of
    (return_path, idx) keys that maximizes total relevance subject to:
      - at most max_articles total
      - equal UP-majority and DOWN-majority counts (strict balance)
    """
    half = max_articles // 2
    ups = sorted([c for c in candidates if c[3] == "UP"], key=lambda c: -c[2])
    downs = sorted([c for c in candidates if c[3] == "DOWN"], key=lambda c: -c[2])

    take_per_side = min(half, len(ups), len(downs))
    selected = set()
    for c in ups[:take_per_side]:
        selected.add((c[0], c[1]))
    for c in downs[:take_per_side]:
        selected.add((c[0], c[1]))
    return selected


def compute_article_tone(factors_json_str):
    """Classify an article by net factor sentiment: POS / NEG / MIX / None.

    POS if more positive factors than negative, NEG if vice versa, MIX on tie,
    None if no factors / unparseable. Used by 4-way balanced selection so we
    can balance on article tone (the dominant source of input bias) on top of
    the outcome-direction balance.
    """
    if not isinstance(factors_json_str, str) or not factors_json_str.strip():
        return None
    try:
        data = json.loads(factors_json_str)
    except Exception:
        return None
    if isinstance(data, dict):
        factors = data.get("factors", [])
    elif isinstance(data, list):
        factors = data
    else:
        return None
    n_pos = sum(1 for x in factors if isinstance(x, dict) and x.get("direction") == "positive")
    n_neg = sum(1 for x in factors if isinstance(x, dict) and x.get("direction") == "negative")
    if n_pos == 0 and n_neg == 0:
        return None
    if n_pos > n_neg:
        return "POS"
    if n_neg > n_pos:
        return "NEG"
    return "MIX"


def gather_candidates_with_tone(return_path, target_ticker, horizon):
    """Like gather_candidates_for_ticker(horizon=...) but also returns the
    article tone. Returns (path, idx, relevance, direction, tone) 5-tuples.
    MIX-toned and tone-less articles are excluded so the 4-way balance only
    sees directionally-toned articles.
    """
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if not os.path.exists(pred_path):
        return []
    df = pd.read_parquet(pred_path).reset_index(drop=True)

    tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
    candidates = []
    for idx, t in enumerate(tickers_in_df):
        if t != target_ticker:
            continue
        row = df.iloc[idx]
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue
        rel = row.get("relevance")
        if not pd.notna(rel):
            continue
        relevance = float(rel)
        if relevance < RELEVANCE_THRESHOLD:
            continue
        direction = per_horizon_actual_direction(row, horizon)
        if direction is None:
            continue
        tone = compute_article_tone(row.get("factors_json"))
        if tone not in ("POS", "NEG"):
            continue  # exclude MIX and tone-less articles from the 4-way balance
        candidates.append((return_path, idx, relevance, direction, tone))
    return candidates


def select_4way_balanced_top_n(candidates, max_articles=MAX_ARTICLES_PER_TICKER):
    """4-way balanced top-N selection over (tone × outcome) quadrants.

    Candidates are (path, idx, relevance, direction, tone) 5-tuples. For each
    of (POS×UP, POS×DOWN, NEG×UP, NEG×DOWN), take min(max_articles//4,
    smallest quadrant size) rows by relevance descending. Final eval set is
    balanced on BOTH the article-tone axis and the outcome-direction axis.

    Returns a set of (path, idx) keys.
    """
    quadrants = {
        ("POS", "UP"):   [],
        ("POS", "DOWN"): [],
        ("NEG", "UP"):   [],
        ("NEG", "DOWN"): [],
    }
    for c in candidates:
        key = (c[4], c[3])  # (tone, direction)
        if key in quadrants:
            quadrants[key].append(c)
    for q in quadrants.values():
        q.sort(key=lambda c: -c[2])  # by relevance descending

    per_q_cap = max_articles // 4
    take = min(per_q_cap, *(len(q) for q in quadrants.values()))

    selected = set()
    for q in quadrants.values():
        for c in q[:take]:
            selected.add((c[0], c[1]))
    return selected


# ============================================================
# FILE PROCESSING & MAIN
# ============================================================

def _load_or_init_pred_df(return_path):
    """Load the prediction parquet (resume) or the source parquet (fresh)."""
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if os.path.exists(pred_path):
        df = pd.read_parquet(pred_path)
        is_resume = True
    else:
        df = pd.read_parquet(return_path)
        is_resume = False

    # Reset to a 0..N-1 RangeIndex so positional iteration (enumerate) matches
    # label-based writes (df.at[idx, ...]). The source parquets are built by
    # filtering a larger combined dataset without reset_index, leaving label
    # gaps like [364..713] for MSFT — without this reset, df.at[idx,...] writes
    # at non-existent labels and silently creates NaN phantom rows.
    df = df.reset_index(drop=True)

    for horizon in HORIZONS:
        if f"pred_{horizon}" not in df.columns:
            df[f"pred_{horizon}"] = None
    # Per-row metadata columns
    if "relevance" not in df.columns:
        df["relevance"] = None
    if "relevance_reasoning" not in df.columns:
        df["relevance_reasoning"] = None
    if "factors_json" not in df.columns:
        df["factors_json"] = None
    if "selected" not in df.columns:
        df["selected"] = None
    return df, pred_path, is_resume


def _has_factor_extraction(row):
    """True iff this row already has both a relevance score and stored factors."""
    rel = row.get("relevance")
    factors = row.get("factors_json")
    if not pd.notna(rel):
        return False
    if not isinstance(factors, str) or not factors.strip():
        return False
    return True


def extract_relevance_in_file(return_path, target_ticker):
    """Run factor extraction for rows of target_ticker in this file that don't
    have a relevance/factors_json value yet. Writes back to the prediction parquet.
    """
    df, pred_path, _ = _load_or_init_pred_df(return_path)

    tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
    ticker_indices = [i for i, t in enumerate(tickers_in_df) if t == target_ticker]

    todo = []
    for idx in ticker_indices:
        row = df.iloc[idx]
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue
        if _has_factor_extraction(row):
            continue
        todo.append(idx)

    if not todo:
        # Still persist any newly-added columns
        pq.write_table(pa.Table.from_pandas(df), pred_path)
        return

    print(f"  📥 Extracting factors for {target_ticker} in {os.path.basename(return_path)}: {len(todo)} rows")
    for i, idx in enumerate(todo, 1):
        row = df.iloc[idx]
        result = extract_factors_for_row(row)
        if result is None:
            df.at[idx, "relevance"] = 0.0
            df.at[idx, "relevance_reasoning"] = "extraction failed"
            df.at[idx, "factors_json"] = json.dumps({"factors": []})
        else:
            factors_data, relevance, relevance_reasoning = result
            df.at[idx, "relevance"] = float(relevance)
            df.at[idx, "relevance_reasoning"] = relevance_reasoning
            df.at[idx, "factors_json"] = json.dumps(factors_data)

        if i % 10 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)} extracted")
        # Checkpoint after each row
        pq.write_table(pa.Table.from_pandas(df), pred_path)

    pq.write_table(pa.Table.from_pandas(df), pred_path)


def gather_candidates_for_ticker(return_path, target_ticker, horizon=None):
    """Read the prediction parquet and return candidate tuples for selection:
    (return_path, idx, relevance, direction).

    When `horizon` is None, classify rows by 5-horizon-majority direction
    (legacy behavior). When `horizon` is a string in HORIZONS, classify by
    that single horizon's actual_direction column (per-horizon balanced
    selection). Candidates without a resolvable direction or with relevance
    below RELEVANCE_THRESHOLD are filtered out.
    """
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if not os.path.exists(pred_path):
        return []
    df = pd.read_parquet(pred_path).reset_index(drop=True)

    tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
    candidates = []
    for idx, t in enumerate(tickers_in_df):
        if t != target_ticker:
            continue
        row = df.iloc[idx]
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue
        rel = row.get("relevance")
        if not pd.notna(rel):
            continue
        relevance = float(rel)
        if relevance < RELEVANCE_THRESHOLD:
            continue
        if horizon is None:
            direction = majority_actual_direction(row)
        else:
            direction = per_horizon_actual_direction(row, horizon)
        if direction is None:
            continue
        candidates.append((return_path, idx, relevance, direction))
    return candidates


def predict_selected_in_file(return_path, target_ticker, selected_keys):
    """Mark selected/unselected for target_ticker rows in this file, then run
    per-horizon predictions only for the selected rows.
    """
    df, pred_path, _ = _load_or_init_pred_df(return_path)

    tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
    ticker_indices = [i for i, t in enumerate(tickers_in_df) if t == target_ticker]
    if not ticker_indices:
        pq.write_table(pa.Table.from_pandas(df), pred_path)
        return

    # Mark selection state for every ticker row in this file
    for idx in ticker_indices:
        is_selected = (return_path, idx) in selected_keys
        df.at[idx, "selected"] = bool(is_selected)
        # If we previously wrote SKIPPED predictions for this row, clear them so
        # selected rows get re-predicted with the new pipeline.
        for h in HORIZONS:
            pred_json = df.at[idx, f"pred_{h}"]
            if isinstance(pred_json, str) and pred_json.strip():
                try:
                    pred = json.loads(pred_json)
                except Exception:
                    pred = None
                if isinstance(pred, dict) and pred.get("direction") == "SKIPPED":
                    df.at[idx, f"pred_{h}"] = None

    # Build the todo list: selected rows missing any pred_{horizon}
    todo = []
    for idx in ticker_indices:
        if (return_path, idx) not in selected_keys:
            continue
        row = df.iloc[idx]
        all_done = all(
            row.get(f"pred_{h}") is not None
            and isinstance(row.get(f"pred_{h}"), str)
            and row.get(f"pred_{h}").strip() != ""
            for h in HORIZONS
        )
        if not all_done:
            todo.append(idx)

    if not todo:
        pq.write_table(pa.Table.from_pandas(df), pred_path)
        return

    print(f"📈 Predicting {target_ticker} in {os.path.basename(return_path)} — {len(todo)} rows")

    processed = 0
    for idx in todo:
        row = df.iloc[idx]
        article_id = str(row.get("Article_id", ""))
        ticker_str = str(row.get("Stock_symbol", "")).strip()
        date = str(row.get("Date", ""))[:10]
        article_text = clean_article_text(row.get("Article_text", ""))
        relevance = float(row.get("relevance", 0.0))
        relevance_reasoning = str(row.get("relevance_reasoning") or "")

        try:
            factors_data = json.loads(row.get("factors_json") or "{}")
        except Exception:
            factors_data = {"factors": []}

        print(f"  Predicting for {ticker_str} at {date}")
        predictions = predict_horizons_for_row(
            article_id, ticker_str, date, article_text,
            factors_data, relevance, relevance_reasoning,
        )

        for horizon, pred in predictions.items():
            if pred is not None:
                df.at[idx, f"pred_{horizon}"] = json.dumps(pred)

        processed += 1
        print(f"  ✅ {processed}/{len(todo)} done")
        # Checkpoint after each row
        pq.write_table(pa.Table.from_pandas(df), pred_path)

    pq.write_table(pa.Table.from_pandas(df), pred_path)


def predict_file_legacy(return_path, target_ticker=None):
    """Legacy single-prompt pipeline path. No relevance, no balanced selection —
    predicts every ticker row directly.
    """
    df, pred_path, _ = _load_or_init_pred_df(return_path)

    tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
    if target_ticker is not None:
        ticker_indices = [i for i, t in enumerate(tickers_in_df) if t == target_ticker]
    else:
        ticker_indices = list(range(len(df)))

    todo = []
    for idx in ticker_indices:
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

    if not todo:
        return

    label = f"target={target_ticker}" if target_ticker else "all tickers"
    print(f"📈 [legacy] {os.path.basename(return_path)} ({label}) — {len(todo)} rows")

    processed = 0
    for idx in todo:
        row = df.iloc[idx]
        ticker = str(row.get("Stock_symbol", "")).strip()
        date = str(row.get("Date", ""))[:10]
        hist_text = load_historical_prices_legacy(ticker, date)
        predictions = predict_for_row_legacy(row, hist_text)
        for horizon, pred in predictions.items():
            if pred is not None:
                df.at[idx, f"pred_{horizon}"] = json.dumps(pred)
        processed += 1
        print(f"  ✅ {processed}/{len(todo)} done")
        pq.write_table(pa.Table.from_pandas(df), pred_path)

    pq.write_table(pa.Table.from_pandas(df), pred_path)


def write_eval_csv(return_path):
    """Read the current prediction parquet and write the evaluation CSV.

    Rows explicitly deselected by the balanced top-N selection (selected=False)
    are excluded. Rows with selected=True or no selected value (legacy / pre-
    selection state) are included.
    """
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if not os.path.exists(pred_path):
        return
    df = pd.read_parquet(pred_path)

    eval_rows = []
    for _, row in df.iterrows():
        article_text = row.get("Article_text", "")
        if not isinstance(article_text, str) or article_text.strip() == "":
            continue

        # Skip rows explicitly filtered out by balanced selection
        sel = row.get("selected") if "selected" in df.columns else None
        if pd.notna(sel) and not bool(sel):
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


def process_ticker_enhanced(ticker, files_for_ticker):
    """Three-phase pipeline for one ticker:
        1. Factor extraction across all batches (populates relevance, factors_json)
        2. Balanced top-N selection across all batches (sets selected column)
        3. Per-horizon prediction only on selected rows
    Phases 1 and 3 are routed through the Batch API when USE_BATCH_API is True.
    Phase 2 is local (no LLM calls) and unchanged.
    """
    print(f"\n  Phase 1: factor extraction "
          f"({'batch' if USE_BATCH_API else 'sync'})")
    if USE_BATCH_API:
        extract_factors_batch(ticker, files_for_ticker)
    else:
        for f in files_for_ticker:
            return_path = os.path.join(RETURN_DIR, f)
            extract_relevance_in_file(return_path, target_ticker=ticker)

    if PER_HORIZON_BALANCED_SELECTION and FOUR_WAY_TONE_BALANCED_SELECTION:
        print(f"  Phase 2: per-horizon 4-way balanced (tone × outcome), "
              f"cap {MAX_ARTICLES_PER_TICKER}/horizon → union")
        selected_keys = set()
        for h in HORIZONS:
            h_candidates = []
            for f in files_for_ticker:
                return_path = os.path.join(RETURN_DIR, f)
                h_candidates.extend(
                    gather_candidates_with_tone(return_path, ticker, horizon=h)
                )
            q_counts = {
                ("POS", "UP"):   sum(1 for c in h_candidates if c[4] == "POS" and c[3] == "UP"),
                ("POS", "DOWN"): sum(1 for c in h_candidates if c[4] == "POS" and c[3] == "DOWN"),
                ("NEG", "UP"):   sum(1 for c in h_candidates if c[4] == "NEG" and c[3] == "UP"),
                ("NEG", "DOWN"): sum(1 for c in h_candidates if c[4] == "NEG" and c[3] == "DOWN"),
            }
            h_selected = select_4way_balanced_top_n(
                h_candidates, max_articles=MAX_ARTICLES_PER_TICKER
            )
            print(f"    {h:>4}: quadrants "
                  f"P+U={q_counts[('POS','UP')]:>3} "
                  f"P+D={q_counts[('POS','DOWN')]:>3} "
                  f"N+U={q_counts[('NEG','UP')]:>3} "
                  f"N+D={q_counts[('NEG','DOWN')]:>3}  →  "
                  f"selected {len(h_selected):>3} "
                  f"({len(h_selected)//4 if h_selected else 0} per quadrant)")
            selected_keys |= h_selected
        print(f"    Union across horizons: {len(selected_keys)} rows")
    elif PER_HORIZON_BALANCED_SELECTION:
        print(f"  Phase 2: per-horizon balanced top-{MAX_ARTICLES_PER_TICKER} "
              f"× {len(HORIZONS)} horizons → union")
        selected_keys = set()
        for h in HORIZONS:
            h_candidates = []
            for f in files_for_ticker:
                return_path = os.path.join(RETURN_DIR, f)
                h_candidates.extend(
                    gather_candidates_for_ticker(return_path, ticker, horizon=h)
                )
            h_ups = sum(1 for c in h_candidates if c[3] == "UP")
            h_downs = sum(1 for c in h_candidates if c[3] == "DOWN")
            h_selected = select_balanced_top_n(
                h_candidates, max_articles=MAX_ARTICLES_PER_TICKER
            )
            h_sel_up = sum(1 for c in h_candidates
                           if c[3] == "UP" and (c[0], c[1]) in h_selected)
            h_sel_dn = sum(1 for c in h_candidates
                           if c[3] == "DOWN" and (c[0], c[1]) in h_selected)
            print(f"    {h:>4}: candidates {len(h_candidates):>4} "
                  f"(UP={h_ups}, DOWN={h_downs})  →  "
                  f"selected {len(h_selected):>3} (UP={h_sel_up}, DOWN={h_sel_dn})")
            selected_keys |= h_selected
        print(f"    Union across horizons: {len(selected_keys)} rows")
    else:
        print(f"  Phase 2: 5-horizon-majority balanced top-{MAX_ARTICLES_PER_TICKER} selection")
        all_candidates = []
        for f in files_for_ticker:
            return_path = os.path.join(RETURN_DIR, f)
            all_candidates.extend(gather_candidates_for_ticker(return_path, ticker))

        total_up = sum(1 for c in all_candidates if c[3] == "UP")
        total_down = sum(1 for c in all_candidates if c[3] == "DOWN")
        selected_keys = select_balanced_top_n(
            all_candidates, max_articles=MAX_ARTICLES_PER_TICKER
        )
        n_up = sum(1 for c in all_candidates if c[3] == "UP" and (c[0], c[1]) in selected_keys)
        n_down = sum(1 for c in all_candidates if c[3] == "DOWN" and (c[0], c[1]) in selected_keys)
        print(f"    Candidates: {len(all_candidates)} (UP={total_up}, DOWN={total_down})")
        print(f"    Selected:   {len(selected_keys)} (UP={n_up}, DOWN={n_down})")

    print(f"  Phase 3: per-horizon predictions for {len(selected_keys)} rows "
          f"({'batch' if USE_BATCH_API else 'sync'})")
    if USE_BATCH_API:
        predict_horizons_batch(ticker, files_for_ticker, selected_keys)
    else:
        for f in files_for_ticker:
            return_path = os.path.join(RETURN_DIR, f)
            predict_selected_in_file(return_path, ticker, selected_keys)


def main():
    files = sorted(f for f in os.listdir(RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No return batch parquet files found.")
        return

    pipeline_label = "enhanced" if USE_ENHANCED_PIPELINE else "legacy"
    transport_label = " + batch API" if (USE_ENHANCED_PIPELINE and USE_BATCH_API) else ""
    print(f"Pipeline: {pipeline_label}{transport_label}")

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
        print(f"\n🎯 [{i}/{len(ticker_order)}] Ticker: {ticker}")
        if USE_ENHANCED_PIPELINE:
            process_ticker_enhanced(ticker, ticker_files[ticker])
        else:
            for f in ticker_files[ticker]:
                return_path = os.path.join(RETURN_DIR, f)
                predict_file_legacy(return_path, target_ticker=ticker)

    # Phase 2: write eval CSVs once at the end.
    print("\n📄 Writing evaluation CSVs...")
    for f in files:
        write_eval_csv(os.path.join(RETURN_DIR, f))


if __name__ == "__main__":
    main()
