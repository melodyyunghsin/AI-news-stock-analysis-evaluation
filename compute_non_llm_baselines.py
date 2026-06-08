"""Non-LLM baselines for direction prediction.

Two baselines, scored on the same 4-way balanced eval set as the LLM pipeline:

1. **FinBERT** — `ProsusAI/finbert`, financial-news sentiment classifier.
   Predict UP if P(positive) > P(negative), else DOWN.
2. **Momentum** — predict UP if the 20-day return prior to the article date was
   positive, else DOWN. Pure price signal; no article content used.

Output:
    data/balanced_focused_evaluation_summary_gemini_k5/baselines_non_llm.csv

Tells you whether the LLM pipeline meaningfully beats simpler approaches.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reuse cleaning + filter logic from gemini_predict
import gemini_predict as gp

# ============================================================
# CONFIG (match the production pipeline)
# ============================================================

PRED_DIR = "data/balanced_focused_predictions_gemini_k5"
PRICE_DIR = "data/full_history"
OUT_DIR = "data/balanced_focused_evaluation_summary_gemini_k5"
HORIZONS = ["1d", "3d", "5d", "10d", "21d"]
RELEVANCE_THRESHOLD = 0.0
MAX_PER_TICKER = 100
MOMENTUM_WINDOW = 20  # days of return lookback for the momentum baseline
FINBERT_MODEL = "ProsusAI/finbert"


# ============================================================
# 4-WAY BALANCED FILTER (mirrors merge_evaluation._apply_four_way_filter)
# ============================================================

def _tone_from_factors_json(fj):
    if not isinstance(fj, str) or not fj.strip():
        return None
    try:
        data = json.loads(fj)
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
    if n_pos > n_neg: return "POS"
    if n_neg > n_pos: return "NEG"
    return None


def four_way_balanced_filter(df, horizon):
    """Return the 4-way balanced subset for one horizon.

    Mirrors merge_evaluation._apply_four_way_filter so the baselines are
    scored on EXACTLY the same articles that the LLM pipeline was scored on.
    """
    df = df.copy()
    df["_act_dir"] = df[f"actual_direction_{horizon}"].astype(str).str.strip().str.upper()
    df["_pred_rel"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["Article_Tone"] = df["factors_json"].apply(_tone_from_factors_json)

    # Match the merge_evaluation step that drops rows without an LLM prediction —
    # we want to score baselines on the same article set the LLM was scored on.
    df = df.dropna(subset=[f"pred_{horizon}"])
    df = df[df[f"pred_{horizon}"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    df = df.dropna(subset=["_pred_rel"])
    df = df[df["_act_dir"].isin(["UP", "DOWN"])]
    df = df[df["Article_Tone"].isin(["POS", "NEG"])]
    df = df[df["_pred_rel"] >= RELEVANCE_THRESHOLD]

    per_q_cap = MAX_PER_TICKER // 4
    keep_idx = []
    for _, group in df.groupby("Stock_symbol", sort=False):
        quadrants = {}
        for tone in ("POS", "NEG"):
            for outcome in ("UP", "DOWN"):
                q = group[
                    (group["Article_Tone"] == tone)
                    & (group["_act_dir"] == outcome)
                ].sort_values("_pred_rel", ascending=False)
                quadrants[(tone, outcome)] = q
        take = min(per_q_cap, *(len(q) for q in quadrants.values()))
        if take == 0:
            continue
        for q in quadrants.values():
            keep_idx.extend(q.head(take).index.tolist())
    return df.loc[keep_idx]


# ============================================================
# BASELINE 1: FinBERT
# ============================================================

def compute_finbert(eval_df):
    """Run FinBERT on every Article_text. Returns Series of 'UP'/'DOWN'/None.

    Uses gp.clean_article_text to strip FNSPID boilerplate before tokenizing.
    Truncates to FinBERT's 512-token limit (first ~2000 chars after cleaning).
    """
    print(f"📥 Loading {FINBERT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"  Device: {device}  |  Labels: {id2label}")

    # Find which label index corresponds to "positive" and "negative"
    pos_idx = next(k for k, v in id2label.items() if v.lower() == "positive")
    neg_idx = next(k for k, v in id2label.items() if v.lower() == "negative")

    predictions = []
    n = len(eval_df)
    for i, (_idx, row) in enumerate(eval_df.iterrows()):
        text = row.get("Article_text", "")
        if not isinstance(text, str) or not text.strip():
            predictions.append(None)
            continue
        cleaned = gp.clean_article_text(text)
        if len(cleaned) < 50:
            predictions.append(None)
            continue
        inputs = tokenizer(
            cleaned, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=0)
        # Binary: stronger of {positive, negative} — ignore neutral
        predictions.append("UP" if probs[pos_idx] > probs[neg_idx] else "DOWN")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n} articles scored")

    print(f"  Done — {sum(1 for p in predictions if p)} / {n} scored")
    return predictions


# ============================================================
# BASELINE 2: Price momentum
# ============================================================

def compute_momentum(eval_df, window=MOMENTUM_WINDOW):
    """Predict UP if `window`-day return before article date was positive.

    Pure price signal; the article itself is not used.
    """
    predictions = []
    price_cache = {}
    for _, row in eval_df.iterrows():
        ticker = str(row["Stock_symbol"]).strip()
        try:
            date = pd.to_datetime(str(row["Date"])[:10])
        except Exception:
            predictions.append(None)
            continue
        if ticker not in price_cache:
            path = os.path.join(PRICE_DIR, f"{ticker}.csv")
            if not os.path.exists(path):
                price_cache[ticker] = None
            else:
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                price_cache[ticker] = df
        prices = price_cache[ticker]
        if prices is None:
            predictions.append(None)
            continue
        hist = prices[prices["date"] < date]
        if len(hist) < window + 1:
            predictions.append(None)
            continue
        p_now = float(hist.iloc[-1]["close"])
        p_then = float(hist.iloc[-window - 1]["close"])
        if p_then <= 0:
            predictions.append(None)
            continue
        ret = (p_now - p_then) / p_then
        predictions.append("UP" if ret > 0 else "DOWN")
    return predictions


# ============================================================
# SCORING
# ============================================================

def bootstrap_mcc_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% CI for MCC. Matches merge_evaluation._bootstrap_mcc_ci."""
    if len(y_true) < 10:
        return (np.nan, np.nan)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    mccs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
            yt = y_true_arr[idx]
            yp = y_pred_arr[idx]
            if len(set(yt)) < 2 or len(set(yp)) < 2:
                continue
            mccs.append(matthews_corrcoef(yt, yp))
    if len(mccs) < 100:
        return (np.nan, np.nan)
    alpha = (1 - ci) / 2
    return (float(np.quantile(mccs, alpha)),
            float(np.quantile(mccs, 1 - alpha)))


def score_baseline(eval_df, pred_col, horizon):
    """Score one baseline on the 4-way filtered eval set for one horizon."""
    df_h = four_way_balanced_filter(eval_df, horizon)
    if df_h.empty:
        return None
    sub = df_h.dropna(subset=[pred_col])
    sub = sub[sub[pred_col].isin(["UP", "DOWN"])]
    if len(sub) < 10:
        return None

    y_true = sub[f"actual_direction_{horizon}"].astype(str).str.upper().tolist()
    y_pred = sub[pred_col].tolist()

    pred_up_rate = float(np.mean(np.array(y_pred) == "UP"))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
            mcc = np.nan
        else:
            mcc = matthews_corrcoef(y_true, y_pred)
        if len(set(y_pred)) >= 2:
            wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            wf1 = np.nan
    ci_lo, ci_hi = bootstrap_mcc_ci(y_true, y_pred)
    return {
        "horizon": horizon,
        "samples": len(sub),
        "pred_up_rate": pred_up_rate,
        "accuracy": acc,
        "wf1": wf1,
        "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
        "mcc_ci_low": ci_lo,
        "mcc_ci_high": ci_hi,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Non-LLM baselines: FinBERT + Momentum")
    print("=" * 70)

    print("\n📥 Loading prediction parquets...")
    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".parquet"))
    full_df = pd.concat(
        [pd.read_parquet(os.path.join(PRED_DIR, f)).reset_index(drop=True) for f in files],
        ignore_index=True,
    )
    print(f"  Total rows: {len(full_df)}")

    print("\n🤖 Computing FinBERT predictions (once for all rows)...")
    full_df["_finbert"] = compute_finbert(full_df)

    print("\n📈 Computing momentum predictions...")
    full_df["_momentum"] = compute_momentum(full_df, window=MOMENTUM_WINDOW)
    n_mom = sum(1 for p in full_df["_momentum"] if p)
    print(f"  Done — {n_mom} / {len(full_df)} scored")

    print("\n📊 Scoring baselines per horizon (4-way balanced filter)...")
    rows = []
    for h in HORIZONS:
        for baseline_col, baseline_name in [("_finbert", "finbert"),
                                             ("_momentum", f"momentum_{MOMENTUM_WINDOW}d")]:
            result = score_baseline(full_df, baseline_col, h)
            if result is None:
                continue
            result["baseline"] = baseline_name
            rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df = out_df[["baseline", "horizon", "samples", "pred_up_rate",
                     "accuracy", "wf1", "mcc", "mcc_ci_low", "mcc_ci_high"]]
    out_path = os.path.join(OUT_DIR, "baselines_non_llm.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Wrote {out_path}\n")

    # Pretty-print comparison
    print("=" * 90)
    print(f"{'Baseline':<18}{'Horizon':<8}{'Samples':>8}{'PredUP%':>9}{'Accuracy':>10}{'MCC':>9}{'MCC 95% CI':>22}")
    print("-" * 90)
    for _, r in out_df.iterrows():
        ci_str = f"[{r['mcc_ci_low']:+.3f}, {r['mcc_ci_high']:+.3f}]" if pd.notna(r['mcc_ci_low']) else "n/a"
        print(f"{r['baseline']:<18}{r['horizon']:<8}{int(r['samples']):>8}"
              f"{r['pred_up_rate']*100:>8.1f}%{r['accuracy']:>10.4f}{r['mcc']:>+9.4f}{ci_str:>22}")
    print("=" * 90)

    # Quick LLM-vs-baseline summary
    try:
        llm = pd.read_csv(os.path.join(OUT_DIR, "overall_metrics_by_horizon.csv"))
        print("\n📌 LLM pipeline vs non-LLM baselines (MCC):")
        print(f"{'Horizon':<8}{'LLM':>12}{'FinBERT':>12}{'Momentum':>12}{'LLM−FinBERT':>14}{'LLM−Momentum':>14}")
        for h in HORIZONS:
            llm_mcc = float(llm[llm["horizon"] == h]["mcc"].iloc[0])
            fin = out_df[(out_df["baseline"] == "finbert") & (out_df["horizon"] == h)]
            mom = out_df[(out_df["baseline"] == f"momentum_{MOMENTUM_WINDOW}d") & (out_df["horizon"] == h)]
            fin_mcc = float(fin["mcc"].iloc[0]) if not fin.empty else np.nan
            mom_mcc = float(mom["mcc"].iloc[0]) if not mom.empty else np.nan
            print(f"{h:<8}{llm_mcc:>+12.4f}{fin_mcc:>+12.4f}{mom_mcc:>+12.4f}"
                  f"{llm_mcc-fin_mcc:>+14.4f}{llm_mcc-mom_mcc:>+14.4f}")
    except Exception as e:
        print(f"⚠️ Couldn't compare to LLM overall_metrics: {e}")


if __name__ == "__main__":
    main()
