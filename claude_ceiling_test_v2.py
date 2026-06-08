"""Claude ceiling test v2 — runs on balanced_focused_dataset.

Mirrors the gemini_predict.py pipeline but swaps in the Anthropic Messages
Batches API. Output parquets are in the same schema as gemini_predict.py,
so merge_evaluation_claude.py (a thin wrapper) can score them with the
same code.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python claude_ceiling_test_v2.py

Defaults — AAPL only, MAX=40, model = claude-opus-4-8 (newest), K=1.

Why K=1 on Opus 4.8? Opus 4.7+ removed the `temperature` parameter
(returns 400 if sent). K-sample self-consistency requires temperature
variation to produce divergent samples — without it, K calls return
the same answer, so K=1 is the honest setting. This makes the test
HARDER for Claude (no vote-based denoising), which is fine: if Opus
K=1 beats gemini-flash-lite K=5, that's strong evidence the model
matters.

Cheaper / fairer alternatives — edit MODEL and K_SAMPLES at the top:
    MODEL = "claude-sonnet-4-6", K_SAMPLES = 5    →  matches gemini K=5 exactly, ~$8
    MODEL = "claude-haiku-4-5",  K_SAMPLES = 5    →  cheapest, ~$3
    MODEL = "claude-opus-4-8",   K_SAMPLES = 1    →  default; ceiling test, ~$14

Cost estimate (claude-opus-4-8, AAPL only, K=1, with Batch API 50% off):
    Phase 1: ~80 factor extractions
    Phase 3: ~50 selected × 5 horizons × 1 = ~250 prediction calls
    Total: ~330 calls × (~3000 in + ~250 out tokens avg)
    Input:  ~1M tokens × $2.50/M  = ~$2.50
    Output: ~0.08M tokens × $12.50/M = ~$1
    Plus factor-extraction tokens: roughly doubles the figure
    Total: ~$5-10 (well under estimate; spend a few credits then re-evaluate)
"""

import os
import json
import time

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Reuse prompt builders, cleaners, and selection logic from the gemini pipeline.
# We monkey-patch PRED_DIR below so gp's helpers read/write Claude's parquets.
import gemini_predict as gp

# ============================================================
# CONFIG
# ============================================================

MODEL = "claude-opus-4-8"
TICKER_PRIORITY = ["AAPL"]  # start with one ticker
MAX_ARTICLES_PER_TICKER = 40
K_SAMPLES = 1   # see header for the Opus-vs-temperature rationale
SAMPLE_TEMPERATURE = 0.8  # ignored when K_SAMPLES=1 or on Opus 4.7+

PRED_DIR = f"data/claude_test_predictions_{MODEL.replace('-', '_').replace('.', '_')}"
EVAL_DIR = f"data/claude_test_evaluation_results_{MODEL.replace('-', '_').replace('.', '_')}"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

POLL_INTERVAL_SEC = 30

# Redirect gemini_predict's PRED_DIR so its selection helpers read Claude's parquets.
# This is intentional and stays in effect for the entire script run.
gp.PRED_DIR = PRED_DIR
gp.EVAL_DIR = EVAL_DIR

# ============================================================
# CLAUDE BATCH API
# ============================================================

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# Anthropic Batch API restricts custom_id to ^[a-zA-Z0-9_-]{1,64}$
# (no | or . allowed). Build/parse IDs through these helpers.
def _file_stem(return_path):
    return os.path.basename(return_path).replace(".parquet", "")


def _factor_id(rp, idx):
    return f"f_{_file_stem(rp)}_{idx}"


def _pred_id(rp, idx, horizon, k):
    return f"p_{_file_stem(rp)}_{idx}_{horizon}_k{k}"


def _supports_temperature(model: str) -> bool:
    """Opus 4.7 and 4.8 reject temperature/top_p/top_k. Sonnet 4.6 / Haiku 4.5 accept them."""
    return not (model.startswith("claude-opus-4-7") or model.startswith("claude-opus-4-8"))


def build_claude_batch_request(custom_id, prompt, temperature=None, max_tokens=2048):
    """Build one Anthropic batch request entry."""
    params = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None and _supports_temperature(MODEL):
        params["temperature"] = float(temperature)
    return {"custom_id": custom_id, "params": params}


def submit_claude_batch(batch_requests):
    """Submit a batch of requests to Claude. Returns (batch_id, ordered_custom_ids)."""
    if not batch_requests:
        return None, []

    requests_formatted = [
        Request(
            custom_id=r["custom_id"],
            params=MessageCreateParamsNonStreaming(**r["params"]),
        )
        for r in batch_requests
    ]

    batch = client.messages.batches.create(requests=requests_formatted)
    print(f"  📤 Claude batch submitted: {len(batch_requests)} reqs → {batch.id}")
    return batch.id, [r["custom_id"] for r in batch_requests]


def poll_claude_batch(batch_id, ordered_custom_ids):
    """Poll until batch ends. Returns {custom_id: response_text_or_None}."""
    if batch_id is None or not ordered_custom_ids:
        return {k: None for k in ordered_custom_ids}

    elapsed = 0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"  ⏳ {batch_id}: status={batch.processing_status} "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored} (elapsed {elapsed}s)")
        time.sleep(POLL_INTERVAL_SEC)
        elapsed += POLL_INTERVAL_SEC

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            results[result.custom_id] = text
        else:
            results[result.custom_id] = None
            if result.result.type == "errored":
                err = result.result.error
                print(f"  ⚠️ {result.custom_id} errored: {err.type}")

    for cid in ordered_custom_ids:
        results.setdefault(cid, None)

    ok = sum(1 for v in results.values() if v)
    print(f"  ✅ Batch {batch_id} done — {ok}/{len(ordered_custom_ids)} OK")
    return results


# ============================================================
# PIPELINE
# ============================================================

def _load_or_init_pred_df(return_path):
    pred_path = os.path.join(PRED_DIR, os.path.basename(return_path))
    if os.path.exists(pred_path):
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_parquet(return_path)
    df = df.reset_index(drop=True)
    for horizon in gp.HORIZONS:
        if f"pred_{horizon}" not in df.columns:
            df[f"pred_{horizon}"] = None
    for col in ("relevance", "relevance_reasoning", "factors_json", "selected"):
        if col not in df.columns:
            df[col] = None
    return df, pred_path


def extract_factors_phase(ticker, files_for_ticker):
    file_dfs, file_paths, plan = {}, {}, []

    for f in files_for_ticker:
        return_path = os.path.join(gp.RETURN_DIR, f)
        df, pred_path = _load_or_init_pred_df(return_path)
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
            if pd.notna(row.get("relevance")) and isinstance(row.get("factors_json"), str):
                continue
            cleaned = gp.clean_article_text(article_text)
            if len(cleaned) < 150:
                continue
            date = str(row.get("Date", ""))[:10]
            company_context = gp.get_company_context(ticker)
            prompt = gp.build_factor_extraction_prompt(ticker, cleaned, company_context, date)
            plan.append((return_path, idx, prompt))

    if not plan:
        for return_path, df in file_dfs.items():
            pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
        return

    print(f"  📥 Factor extraction for {ticker}: {len(plan)} rows")
    batch_requests = [
        build_claude_batch_request(
            custom_id=_factor_id(rp, idx),
            prompt=prompt,
            max_tokens=1500,
        )
        for (rp, idx, prompt) in plan
    ]
    batch_id, ordered_keys = submit_claude_batch(batch_requests)
    results = poll_claude_batch(batch_id, ordered_keys)

    for (rp, idx, _prompt) in plan:
        key = _factor_id(rp, idx)
        raw = results.get(key)
        clean = gp.clean_raw_output(raw) if raw else None
        parsed = gp.extract_json(clean) if clean else None

        df = file_dfs[rp]
        if isinstance(parsed, dict) and "factors" in parsed:
            relevance = max(0.0, min(1.0, float(parsed.get("relevance", 0.0))))
            reasoning = str(parsed.get("relevance_reasoning", ""))
            df.at[idx, "relevance"] = relevance
            df.at[idx, "relevance_reasoning"] = reasoning
            df.at[idx, "factors_json"] = json.dumps(parsed)
        elif isinstance(parsed, list):
            wrapped = {"relevance": 0.5, "relevance_reasoning": "unknown", "factors": parsed}
            df.at[idx, "relevance"] = 0.5
            df.at[idx, "relevance_reasoning"] = "unknown"
            df.at[idx, "factors_json"] = json.dumps(wrapped)
        else:
            df.at[idx, "relevance"] = 0.0
            df.at[idx, "relevance_reasoning"] = "extraction failed"
            df.at[idx, "factors_json"] = json.dumps({"factors": []})

    for return_path, df in file_dfs.items():
        pq.write_table(pa.Table.from_pandas(df), file_paths[return_path])
    print(f"  💾 Factor results saved for {ticker}")


def predict_phase(ticker, files_for_ticker, selected_keys):
    file_dfs, file_paths, plan = {}, {}, []

    for f in files_for_ticker:
        return_path = os.path.join(gp.RETURN_DIR, f)
        df, pred_path = _load_or_init_pred_df(return_path)
        file_dfs[return_path] = df
        file_paths[return_path] = pred_path

        tickers_in_df = df["Stock_symbol"].astype(str).str.strip()
        ticker_indices = [i for i, t in enumerate(tickers_in_df) if t == ticker]
        if not ticker_indices:
            continue

        for idx in ticker_indices:
            is_selected = (return_path, idx) in selected_keys
            df.at[idx, "selected"] = bool(is_selected)
            if not is_selected:
                continue

            row = df.iloc[idx]
            article_id = str(row.get("Article_id", ""))
            date = str(row.get("Date", ""))[:10]
            article_text = gp.clean_article_text(row.get("Article_text", ""))
            relevance = float(row.get("relevance", 0.0))
            relevance_reasoning = str(row.get("relevance_reasoning") or "")

            try:
                factors_data = json.loads(row.get("factors_json") or "{}")
            except Exception:
                factors_data = {"factors": []}

            company_context = gp.get_company_context(ticker)
            price_summary = gp.build_price_summary(ticker, date)
            factors_text = gp.format_factors_for_prediction(factors_data)

            for horizon in gp.HORIZONS:
                pred_json = df.at[idx, f"pred_{horizon}"]
                if isinstance(pred_json, str) and pred_json.strip():
                    continue
                prompt = gp.build_prediction_prompt(
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
            custom_id = _pred_id(rp, idx, horizon, k)
            batch_requests.append(
                build_claude_batch_request(custom_id, prompt, temperature=sample_temp, max_tokens=600)
            )

    n_effective_k = K_SAMPLES if _supports_temperature(MODEL) else 1
    if K_SAMPLES > 1 and not _supports_temperature(MODEL):
        print(f"  ⚠️ {MODEL} doesn't accept temperature; K samples will be identical. "
              f"Submitting K={K_SAMPLES} anyway for vote-counting structure.")

    print(f"  📤 Predictions for {ticker}: {len(plan)} (row,horizon) × K={K_SAMPLES} "
          f"= {len(batch_requests)} requests")

    batch_id, ordered_keys = submit_claude_batch(batch_requests)
    results = poll_claude_batch(batch_id, ordered_keys)

    for (rp, idx, horizon, _prompt) in plan:
        samples = []
        for k in range(K_SAMPLES):
            custom_id = _pred_id(rp, idx, horizon, k)
            raw = results.get(custom_id)
            clean = gp.clean_raw_output(raw) if raw else None
            parsed = gp.extract_json(clean) if clean else None
            if isinstance(parsed, dict) and parsed.get("direction") in ("UP", "DOWN"):
                samples.append(parsed)

        df = file_dfs[rp]
        if not samples:
            continue

        if len(samples) == 1:
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
    print(f"  💾 Predictions saved for {ticker}")


def process_ticker(ticker, files_for_ticker):
    print(f"\n  Phase 1: factor extraction (Claude batch)")
    extract_factors_phase(ticker, files_for_ticker)

    print(f"  Phase 2: per-horizon 4-way balanced selection")
    selected_keys = set()
    for h in gp.HORIZONS:
        h_candidates = []
        for f in files_for_ticker:
            return_path = os.path.join(gp.RETURN_DIR, f)
            h_candidates.extend(gp.gather_candidates_with_tone(return_path, ticker, horizon=h))
        q_counts = {
            ("POS", "UP"):   sum(1 for c in h_candidates if c[4] == "POS" and c[3] == "UP"),
            ("POS", "DOWN"): sum(1 for c in h_candidates if c[4] == "POS" and c[3] == "DOWN"),
            ("NEG", "UP"):   sum(1 for c in h_candidates if c[4] == "NEG" and c[3] == "UP"),
            ("NEG", "DOWN"): sum(1 for c in h_candidates if c[4] == "NEG" and c[3] == "DOWN"),
        }
        h_selected = gp.select_4way_balanced_top_n(h_candidates, max_articles=MAX_ARTICLES_PER_TICKER)
        print(f"    {h:>4}: quadrants "
              f"P+U={q_counts[('POS','UP')]:>3} "
              f"P+D={q_counts[('POS','DOWN')]:>3} "
              f"N+U={q_counts[('NEG','UP')]:>3} "
              f"N+D={q_counts[('NEG','DOWN')]:>3}  →  "
              f"selected {len(h_selected):>3}")
        selected_keys |= h_selected
    print(f"    Union across horizons: {len(selected_keys)} rows")

    print(f"  Phase 3: predictions (Claude batch)")
    predict_phase(ticker, files_for_ticker, selected_keys)


def main():
    print("=" * 70)
    print(f"CLAUDE CEILING TEST v2")
    print(f"  Model:       {MODEL}")
    print(f"  Tickers:     {TICKER_PRIORITY}")
    print(f"  Max/ticker:  {MAX_ARTICLES_PER_TICKER} (4-way balanced, 10/quadrant)")
    print(f"  K_SAMPLES:   {K_SAMPLES}")
    print(f"  Temperature: {'supported' if _supports_temperature(MODEL) else 'NOT SUPPORTED — K samples identical'}")
    print(f"  PRED_DIR:    {PRED_DIR}")
    print(f"  EVAL_DIR:    {EVAL_DIR}")
    print("=" * 70)

    files = sorted(f for f in os.listdir(gp.RETURN_DIR) if f.endswith(".parquet"))
    if not files:
        print("No source parquets found.")
        return

    ticker_files = gp.index_files_by_ticker(files)
    ticker_order = [t for t in TICKER_PRIORITY if t in ticker_files]
    missing = [t for t in TICKER_PRIORITY if t not in ticker_files]
    if missing:
        print(f"⚠️  Requested tickers not in source: {missing}")

    for i, ticker in enumerate(ticker_order, 1):
        print(f"\n🎯 [{i}/{len(ticker_order)}] Ticker: {ticker}")
        process_ticker(ticker, ticker_files[ticker])

    # Eval CSVs
    print("\n📄 Writing eval CSVs...")
    requested_files = sorted({f for t in ticker_order for f in ticker_files[t]})
    for f in requested_files:
        gp.write_eval_csv(os.path.join(gp.RETURN_DIR, f))


if __name__ == "__main__":
    main()
