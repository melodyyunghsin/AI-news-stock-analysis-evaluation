# Evaluation v2 — News-to-Stock-Direction LLM Benchmark

Pipeline that scores how well three LLMs (local Qwen-7B via Ollama, Google
Gemini, and Anthropic Claude) can predict short-horizon stock direction from
financial news headlines + bodies, evaluated against actual returns at
1d / 3d / 5d / 10d / 21d horizons.

> **Related project:** the browser extension that consumes these models in
> production lives at
> [melodyyunghsin/AI-news-stock-analysis](https://github.com/melodyyunghsin/AI-news-stock-analysis).
> This repo is the offline benchmark used to pick which model the extension
> ships with.

## What's in this repo

```
Evaluation_v2/
├── Prediction
│   ├── qwen_predict.py            # Local Qwen-7B via Ollama (primary model)
│   ├── gemini_predict.py          # Google Gemini (multi-key round-robin)
│   └── claude_ceiling_test.py     # Claude frontier-model ceiling baseline
│
├── Data prep
│   ├── audit_articles.py          # Quality audit on raw return_batches
│   ├── clean_articles.py          # Strip Nasdaq scraper boilerplate
│   ├── reclassify_returns.py      # Re-bucket UP/DOWN strength thresholds
│   └── compute_returns_50_with_horizon.py
│                                  # Compute actual returns from full_history CSVs
│
├── Focused-eval pipeline
│   ├── build_focused_dataset.py   # Top-relevance article selection per ticker
│   └── diagnose_focused_dataset.py
│
├── Backfills (additive schema migrations)
│   ├── backfill_majority_direction.py
│   └── backfill_relevance_column.py
│
├── Evaluation
│   ├── merge_evaluation.py        # Sweeps confidence/relevance/magnitude thresholds
│   ├── confusion_by_ticker.py     # Per-ticker confusion matrices
│   └── diagnose.py                # Accuracy-by-relevance breakdown
│
└── data/
    ├── evaluation_summary/        # Gemini run report artifacts (committed)
    └── evaluation_summary_qwen/   # Qwen  run report artifacts (committed)
```

Everything else under `data/` (raw articles, price history, per-batch
predictions) is git-ignored — see [`.gitignore`](.gitignore).

## Setup

```bash
python3 -m venv evaluation_v2_venv
source evaluation_v2_venv/bin/activate
pip install -r requirements.txt
```

Copy the env template and fill in your keys:

```bash
cp .env.example .env
# edit .env with your keys, then:
export $(grep -v '^#' .env | xargs)
```

For Qwen, you also need [Ollama](https://ollama.com/) running locally with the
`qwen2.5:7b` model pulled:

```bash
ollama pull qwen2.5:7b
ollama serve   # default port 11434
```

## Pipeline order

The full end-to-end flow, assuming you have `data/full_history/*.csv` price
data and raw `data/article_batches/*.parquet` ready:

```
1. audit_articles.py              → flags broken / paywall / too-short articles
2. clean_articles.py              → writes data/return_batches_clean/
3. compute_returns_50_with_horizon.py
                                  → adds actual_return_{h}, actual_direction_{h}
4. qwen_predict.py                → writes data/prediction_batches_qwen/
   (and/or) gemini_predict.py     → writes data/prediction_batches/
   (and/or) claude_ceiling_test.py
5. backfill_majority_direction.py → adds actual_majority_direction_{h}
   backfill_relevance_column.py   → lifts relevance to top-level column
6. merge_evaluation.py            → writes data/evaluation_summary*/
```

The **focused evaluation** branch (self-consistency on top-relevance articles
per ticker) is:

```
build_focused_dataset.py          → writes data/focused_dataset/
diagnose_focused_dataset.py       → sanity check
qwen_predict.py                   → with paths re-pointed at focused_dataset
merge_evaluation.py               → with PRED_DIR re-pointed
```

See the docstring at the top of `build_focused_dataset.py` for the exact
config flips.

## Configuration

Each script has a `CONFIG` block near the top with input/output dir constants
(`RETURN_DIR`, `PRED_DIR`, `EVAL_DIR`, etc.). Edit those directly when
switching between the standard and focused-evaluation runs.

## Reproducing the full dataset

This repo does **not** ship the underlying article corpus (~25 GB) or the
price history dump. To reproduce from scratch you need:

- The [FNSPID Financial News Dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
  (or an equivalent ticker-tagged news parquet)
- Daily OHLCV CSVs per ticker under `data/full_history/{TICKER}.csv` with
  columns `date`, `close`

Then run the pipeline above.

## Models evaluated

| Script | Model | Where it runs |
|---|---|---|
| `qwen_predict.py` | `qwen2.5:7b` | local via Ollama |
| `gemini_predict.py` | `gemini-3.1-flash-lite-preview` | Google API |
| `claude_ceiling_test.py` | `claude-opus-4-7` | Anthropic API |

## Notes on safety

- **Never commit `.env`** — it's in `.gitignore` for a reason.
- The legacy hardcoded `API_KEYS` list in `gemini_predict.py` was migrated to
  `GEMINI_API_KEYS` env var. Rotate any keys that were ever hardcoded.
