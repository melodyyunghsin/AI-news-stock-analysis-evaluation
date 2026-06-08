import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

# ============================================================
# CONFIG
# ============================================================

PRED_DIR = "data/balanced_focused_predictions_gemini_k5"   # source of truth (live parquets)
EVAL_DIR = "data/balanced_focused_evaluation_results_gemini_k5"   # legacy CSV fallback
OUT_DIR = "data/balanced_focused_evaluation_summary_gemini_k5"

os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = ["1d", "3d", "5d", "10d", "21d"]

CONFIDENCE_THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RELEVANCE_THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# |Actual_Return| filter — drops near-zero return labels which are essentially noise.
MAGNITUDE_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
# Calibrated decision threshold on _p_up = vote_up / K_SAMPLES. With K=5,
# only the breakpoints 0.0/0.2/0.4/0.6/0.8/1.0 produce distinct partitions,
# but we sweep a fine grid so the curve is visible. threshold=0.5 = current
# majority-vote behavior (K=5 ties don't exist since K is odd).
CALIBRATION_THRESHOLDS = [round(0.1 * i, 1) for i in range(11)]

VALID_DIRECTIONS = {"UP", "DOWN"}

# Per-horizon eval filter — mirrors PER_HORIZON_BALANCED_SELECTION in gemini_predict.
# When True, each horizon is scored only on the rows that would have been
# selected for THAT horizon's top-N (50 UP + 50 DOWN by relevance), so the
# eval set at each horizon is exactly 50/50. When False, every union row that
# has both a prediction and an actual direction at that horizon is scored
# (legacy behavior; per-horizon distribution can be skewed).
PER_HORIZON_FILTER = True
PER_HORIZON_MAX_ARTICLES = 100   # must match MAX_ARTICLES_PER_TICKER in gemini_predict
PER_HORIZON_REL_THRESHOLD = 0.0  # must match RELEVANCE_THRESHOLD in gemini_predict
# When True, the per-horizon filter further balances on article tone using a
# 4-way (POS×UP, POS×DOWN, NEG×UP, NEG×DOWN) quadrant split — mirroring
# FOUR_WAY_TONE_BALANCED_SELECTION in gemini_predict. Eliminates the input
# tone bias at the cost of a smaller eval set.
FOUR_WAY_TONE_FILTER = True

# ============================================================
# UTILITIES
# ============================================================

def find_horizons(columns):
    """Detect horizons from column names like Pred_Direction_1d."""
    horizons = set()
    for c in columns:
        m = re.match(r"Pred_Direction_(.+)", c)
        if m:
            horizons.add(m.group(1))
    return sorted(horizons, key=lambda h: (len(h), h))


def _row_to_eval_dict(row):
    """Convert one prediction-parquet row into the same shape that
    qwen_predict.write_eval_csv() would produce. Returns None for rows that
    have no article text or no predictions at all (nothing to score)."""
    article_text = row.get("Article_text", "")
    if not isinstance(article_text, str) or article_text.strip() == "":
        return None

    out = {
        "Article_id": row.get("Article_id"),
        "Date": row.get("Date"),
        "Ticker": row.get("Stock_symbol"),
    }

    # Prefer the dedicated top-level columns; fall back to JSON for old parquets.
    rel_col = row.get("relevance")
    reason_col = row.get("relevance_reasoning")
    relevance_val = float(rel_col) if pd.notna(rel_col) else None
    relevance_reason = reason_col if pd.notna(reason_col) else None
    has_any_pred = False

    for h in HORIZONS:
        pred_json = row.get(f"pred_{h}")
        direction = None
        confidence = None
        explanation = None
        if pred_json and isinstance(pred_json, str) and pred_json.strip():
            try:
                pred = json.loads(pred_json)
                direction = pred.get("direction")
                confidence = 0.0 if direction == "SKIPPED" else pred.get("confidence")
                explanation = pred.get("explanation")
                has_any_pred = True
                # Backfill from JSON only if the column didn't supply a value
                if relevance_val is None:
                    relevance_val = pred.get("relevance")
                    relevance_reason = pred.get("relevance_reasoning")
            except Exception:
                pass
        out[f"Pred_Direction_{h}"] = direction
        out[f"Pred_Confidence_{h}"] = confidence
        out[f"Explanation_{h}"] = explanation

    out["Pred_Relevance"] = relevance_val
    out["Relevance_Reasoning"] = relevance_reason

    # Article tone, derived from the factor extraction step. POS = more
    # positive-direction factors than negative, NEG = vice versa, MIX = equal,
    # None when no factors were extracted. Used downstream for stratified
    # reporting (does the model add value on NEG articles where it has to go
    # against the default-UP grain?).
    out["Article_Tone"] = None
    out["N_Pos_Factors"] = None
    out["N_Neg_Factors"] = None
    factors_json = row.get("factors_json")
    if isinstance(factors_json, str) and factors_json.strip():
        try:
            fdata = json.loads(factors_json)
            if isinstance(fdata, dict):
                factors = fdata.get("factors", [])
            elif isinstance(fdata, list):
                factors = fdata
            else:
                factors = []
            n_pos = sum(1 for x in factors if isinstance(x, dict) and x.get("direction") == "positive")
            n_neg = sum(1 for x in factors if isinstance(x, dict) and x.get("direction") == "negative")
            out["N_Pos_Factors"] = n_pos
            out["N_Neg_Factors"] = n_neg
            if n_pos == 0 and n_neg == 0:
                out["Article_Tone"] = None
            elif n_pos > n_neg:
                out["Article_Tone"] = "POS"
            elif n_neg > n_pos:
                out["Article_Tone"] = "NEG"
            else:
                out["Article_Tone"] = "MIX"
        except Exception:
            pass

    for h in HORIZONS:
        out[f"Actual_Return_{h}"] = row.get(f"return_{h}")
        out[f"Actual_Direction_{h}"] = row.get(f"actual_direction_{h}")
        out[f"Actual_Strength_{h}"] = row.get(f"actual_strength_{h}")
        out[f"Actual_Majority_Direction_{h}"] = row.get(f"actual_majority_direction_{h}")
        out[f"Actual_Up_Days_{h}"] = row.get(f"actual_up_days_{h}")
        out[f"Actual_Down_Days_{h}"] = row.get(f"actual_down_days_{h}")

    if not has_any_pred:
        return None
    return out


def load_predictions_from_parquets():
    """Build the eval dataframe live from prediction parquets in PRED_DIR.

    Lets us run metrics at any point during a long prediction run without
    waiting for the post-run eval-CSV pass.
    """
    if not os.path.isdir(PRED_DIR):
        raise RuntimeError(f"Prediction directory not found: {PRED_DIR}")
    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"No prediction parquets found in {PRED_DIR}")

    print(f"📥 Reading {len(files)} prediction parquet(s) from {PRED_DIR}")
    rows = []
    files_with_preds = 0
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(PRED_DIR, f))
        except Exception as e:
            print(f"  ⚠️ Could not read {f}: {e}")
            continue
        any_in_file = False
        for _, row in df.iterrows():
            ev = _row_to_eval_dict(row)
            if ev is not None:
                ev["source_file"] = f
                rows.append(ev)
                any_in_file = True
        if any_in_file:
            files_with_preds += 1

    print(f"  Found predictions in {files_with_preds}/{len(files)} files; "
          f"{len(rows)} scoreable rows total")
    if not rows:
        raise RuntimeError("No predictions found in any parquet — nothing to evaluate.")
    return pd.DataFrame(rows)


def load_all_eval_files():
    """Legacy loader: read pre-built eval CSVs from EVAL_DIR.

    Kept for the case where the user has eval CSVs from a previous run but no
    matching prediction parquets. Not used by main().
    """
    files = sorted(f for f in os.listdir(EVAL_DIR) if f.endswith(".csv"))
    if not files:
        raise RuntimeError(f"No evaluation CSV files found in {EVAL_DIR}")

    dfs = []
    for f in files:
        path = os.path.join(EVAL_DIR, f)
        df = pd.read_csv(path)
        df["source_file"] = f
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def _safe_wf1(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _safe_mcc(y_true, y_pred):
    # MCC is undefined when either side has only one class.
    if len(y_true) == 0:
        return np.nan
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return matthews_corrcoef(y_true, y_pred)


def _bootstrap_mcc_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% CI for MCC. Returns (low, high) percentile bounds.

    Resamples (y_true, y_pred) pairs with replacement n_bootstrap times,
    computes MCC on each resample, and returns the central CI from the
    resulting MCC distribution. Returns (nan, nan) for samples too small
    or degenerate (single class).
    """
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


def prepare_horizon_df(df, horizon):
    """Return a per-horizon view with normalized columns.

    Filters rows where actual direction is missing/invalid, the prediction is
    missing, or the prediction is "SKIPPED" (low-relevance gated articles).
    """
    pred_dir_col = f"Pred_Direction_{horizon}"
    pred_conf_col = f"Pred_Confidence_{horizon}"
    act_dir_col = f"Actual_Direction_{horizon}"
    act_ret_col = f"Actual_Return_{horizon}"
    act_maj_col = f"Actual_Majority_Direction_{horizon}"

    cols = ["Ticker", pred_dir_col, pred_conf_col, act_dir_col]
    if act_ret_col in df.columns:
        cols.append(act_ret_col)
    if act_maj_col in df.columns:
        cols.append(act_maj_col)
    if "Pred_Relevance" in df.columns:
        cols.append("Pred_Relevance")
    if "Article_Tone" in df.columns:
        cols.append("Article_Tone")
    df_h = df[cols].copy()

    df_h = df_h.dropna(subset=[pred_dir_col, act_dir_col])

    df_h["_pred_dir"] = df_h[pred_dir_col].astype(str).str.strip().str.upper()
    df_h["_act_dir"] = df_h[act_dir_col].astype(str).str.strip().str.upper()

    # Filtering to UP/DOWN automatically drops SKIPPED rows
    df_h = df_h[df_h["_act_dir"].isin(VALID_DIRECTIONS)]
    df_h = df_h[df_h["_pred_dir"].isin(VALID_DIRECTIONS)]

    df_h["_pred_conf"] = pd.to_numeric(df_h[pred_conf_col], errors="coerce")

    # P(UP) reconstructed from direction + confidence. For K=5 vote agreement
    # this exactly recovers vote_up / K_SAMPLES (the pipeline stores confidence
    # = max(up, down)/K, signed by the winning direction). For K=1 it's the
    # model's self-rated confidence, which is less reliable but usable.
    df_h["_p_up"] = np.where(
        df_h["_pred_dir"] == "UP",
        df_h["_pred_conf"],
        1.0 - df_h["_pred_conf"],
    )

    if "Pred_Relevance" in df_h.columns:
        df_h["_pred_rel"] = pd.to_numeric(df_h["Pred_Relevance"], errors="coerce")
    else:
        df_h["_pred_rel"] = np.nan

    if act_ret_col in df_h.columns:
        df_h["_act_return"] = pd.to_numeric(df_h[act_ret_col], errors="coerce")
    else:
        df_h["_act_return"] = np.nan

    if act_maj_col in df_h.columns:
        df_h["_act_maj_dir"] = df_h[act_maj_col].astype(str).str.strip().str.upper()
    # else: leave _act_maj_dir absent so callers can detect "not backfilled yet"

    return df_h


def apply_per_horizon_filter(df_h, horizon):
    """Restrict df_h to the rows that would have been picked by the per-horizon
    balanced top-N selection for this horizon.

    Mirrors gemini_predict.select_balanced_top_n: per ticker, take the top
    PER_HORIZON_MAX_ARTICLES // 2 UP-at-h rows and the same number of DOWN-at-h
    rows, ranked by relevance descending. If one side has fewer candidates,
    the other side is capped to match (strict balance). This makes the
    per-horizon eval set exactly 50/50 at h for each ticker.

    Re-derives the selection at eval time so no pipeline rerun is needed. The
    inputs (Pred_Relevance, Actual_Direction_h) are the same ones the pipeline
    used, so this is equivalent to reading a hypothetical `selected_h` column.
    """
    if df_h.empty:
        return df_h
    if "Pred_Relevance" not in df_h.columns:
        # Can't re-derive without a relevance score — leave df_h unchanged.
        return df_h

    if FOUR_WAY_TONE_FILTER and "Article_Tone" in df_h.columns:
        return _apply_four_way_filter(df_h)

    half = PER_HORIZON_MAX_ARTICLES // 2
    keep_idx = []
    for _, group in df_h.groupby("Ticker", sort=False):
        elig = group[
            group["_pred_rel"].notna()
            & (group["_pred_rel"] >= PER_HORIZON_REL_THRESHOLD)
        ]
        ups = elig[elig["_act_dir"] == "UP"].sort_values("_pred_rel", ascending=False)
        downs = elig[elig["_act_dir"] == "DOWN"].sort_values("_pred_rel", ascending=False)
        take = min(half, len(ups), len(downs))
        if take == 0:
            continue
        keep_idx.extend(ups.head(take).index.tolist())
        keep_idx.extend(downs.head(take).index.tolist())

    return df_h.loc[keep_idx]


def _apply_four_way_filter(df_h):
    """Mirror gemini_predict.select_4way_balanced_top_n at eval time.

    Per ticker, partition eligible rows by (Article_Tone × _act_dir) into 4
    quadrants and keep min(MAX//4, smallest quadrant) rows from each, ranked
    by relevance. MIX-toned and tone-less articles are excluded.
    """
    per_q_cap = PER_HORIZON_MAX_ARTICLES // 4
    keep_idx = []
    for _, group in df_h.groupby("Ticker", sort=False):
        elig = group[
            group["_pred_rel"].notna()
            & (group["_pred_rel"] >= PER_HORIZON_REL_THRESHOLD)
            & group["Article_Tone"].isin(["POS", "NEG"])
        ]
        quadrants = {}
        for tone in ("POS", "NEG"):
            for outcome in ("UP", "DOWN"):
                q = elig[
                    (elig["Article_Tone"] == tone)
                    & (elig["_act_dir"] == outcome)
                ].sort_values("_pred_rel", ascending=False)
                quadrants[(tone, outcome)] = q
        take = min(per_q_cap, *(len(q) for q in quadrants.values()))
        if take == 0:
            continue
        for q in quadrants.values():
            keep_idx.extend(q.head(take).index.tolist())
    return df_h.loc[keep_idx]


def count_skipped(df, horizon):
    pred_dir_col = f"Pred_Direction_{horizon}"
    if pred_dir_col not in df.columns:
        return 0
    s = df[pred_dir_col].astype(str).str.strip().str.upper()
    return int((s == "SKIPPED").sum())


def compute_accuracy_by_ticker(df_h, horizon, all_tickers):
    """Per-ticker direction accuracy / WF1 / MCC for one horizon."""
    samples = (
        df_h.groupby("Ticker")
            .size()
            .reindex(all_tickers, fill_value=0)
            .reset_index(name="samples")
    )

    out_cols = ["Ticker", "horizon", "samples",
                "direction_accuracy", "direction_wf1", "direction_mcc"]

    if df_h.empty:
        samples["horizon"] = horizon
        for col in ["direction_accuracy", "direction_wf1", "direction_mcc"]:
            samples[col] = pd.NA
        return samples[out_cols]

    df_h = df_h.copy()
    df_h["direction_correct"] = (df_h["_pred_dir"] == df_h["_act_dir"]).astype(int)

    accuracy = (
        df_h.groupby("Ticker")
            .agg(direction_accuracy=("direction_correct", "mean"))
            .reset_index()
    )

    def _per_ticker_metrics(g):
        y_true = g["_act_dir"].tolist()
        y_pred = g["_pred_dir"].tolist()
        lo, hi = _bootstrap_mcc_ci(y_true, y_pred)
        return pd.Series({
            "direction_wf1": _safe_wf1(y_true, y_pred),
            "direction_mcc": _safe_mcc(y_true, y_pred),
            "direction_mcc_ci_low": lo,
            "direction_mcc_ci_high": hi,
        })

    wf1_mcc = df_h.groupby("Ticker").apply(_per_ticker_metrics).reset_index()

    summary = samples.merge(accuracy, on="Ticker", how="left").merge(wf1_mcc, on="Ticker", how="left")
    summary["horizon"] = horizon
    return summary[out_cols + ["direction_mcc_ci_low", "direction_mcc_ci_high"]]


def compute_overall(df_h, horizon, skipped_count=0, gt_col="_act_dir"):
    """Overall direction metrics. `gt_col` selects the ground-truth column —
    `_act_dir` (return-sign based, default) or `_act_maj_dir` (day-count majority).
    """
    if gt_col not in df_h.columns:
        return None
    sub = df_h[df_h[gt_col].isin(VALID_DIRECTIONS)]
    n = len(sub)
    if n == 0:
        return {"horizon": horizon, "samples": 0, "skipped": skipped_count,
                "accuracy": np.nan, "wf1": np.nan, "mcc": np.nan}

    y_true = sub[gt_col].tolist()
    y_pred = sub["_pred_dir"].tolist()
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    mcc_lo, mcc_hi = _bootstrap_mcc_ci(y_true, y_pred)
    return {
        "horizon": horizon,
        "samples": n,
        "skipped": skipped_count,
        "accuracy": acc,
        "wf1": _safe_wf1(y_true, y_pred),
        "mcc": _safe_mcc(y_true, y_pred),
        "mcc_ci_low": mcc_lo,
        "mcc_ci_high": mcc_hi,
    }


def metrics_by_magnitude_threshold(df_h, horizon, thresholds):
    """Score only rows where |Actual_Return| >= threshold.

    Near-zero returns are essentially noise — neither the model nor a human
    can predict them. Filtering reveals whether the pipeline has signal on
    rows where direction is actually meaningful.
    """
    rows = []
    sub_all = df_h.dropna(subset=["_act_return"])
    denom = len(sub_all)
    for thr in thresholds:
        sub = sub_all[sub_all["_act_return"].abs() >= thr]
        n = len(sub)
        coverage = (n / denom) if denom > 0 else np.nan
        if n == 0:
            rows.append({"horizon": horizon, "threshold": thr, "samples": 0,
                         "coverage": coverage, "accuracy": np.nan,
                         "wf1": np.nan, "mcc": np.nan})
            continue
        y_true = sub["_act_dir"].tolist()
        y_pred = sub["_pred_dir"].tolist()
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rows.append({
            "horizon": horizon,
            "threshold": thr,
            "samples": n,
            "coverage": coverage,
            "accuracy": acc,
            "wf1": _safe_wf1(y_true, y_pred),
            "mcc": _safe_mcc(y_true, y_pred),
        })
    return rows


def compute_baselines(df_h, horizon, gt_col="_act_dir"):
    """Trivial baselines: always-UP, always-DOWN, majority-class.

    `gt_col` selects the ground-truth column — `_act_dir` (return-sign based,
    default) or `_act_maj_dir` (day-count majority).
    """
    if gt_col not in df_h.columns:
        return []
    sub = df_h[df_h[gt_col].isin(VALID_DIRECTIONS)]
    n = len(sub)
    if n == 0:
        return []
    actual = sub[gt_col].values
    up_frac = float((actual == "UP").mean())
    pred_up_frac = float((sub["_pred_dir"] == "UP").mean())

    rows = []
    for label, pred_value in [("always_UP", "UP"), ("always_DOWN", "DOWN")]:
        rows.append({
            "horizon": horizon, "baseline": label, "samples": n,
            "accuracy": float((actual == pred_value).mean()),
            # WF1/MCC undefined when predictions have one class; report nan
            "wf1": np.nan, "mcc": np.nan,
        })
    majority = "UP" if up_frac >= 0.5 else "DOWN"
    rows.append({
        "horizon": horizon, "baseline": f"majority_class[{majority}]",
        "samples": n,
        "accuracy": float(max(up_frac, 1 - up_frac)),
        "wf1": np.nan, "mcc": np.nan,
    })
    # Pipeline result for comparison
    acc = float((sub["_pred_dir"].values == actual).mean())
    rows.append({
        "horizon": horizon, "baseline": "qwen_pipeline", "samples": n,
        "accuracy": acc,
        "wf1": _safe_wf1(actual.tolist(), sub["_pred_dir"].tolist()),
        "mcc": _safe_mcc(actual.tolist(), sub["_pred_dir"].tolist()),
    })
    # Distribution row for context
    rows.append({
        "horizon": horizon, "baseline": "_class_dist",
        "samples": n, "accuracy": np.nan, "wf1": np.nan, "mcc": np.nan,
        "actual_up_frac": up_frac, "pred_up_frac": pred_up_frac,
    })
    return rows


def metrics_by_article_tone(df_h, horizon):
    """Stratify metrics by article tone (POS/NEG/MIX) from factor extraction.

    The NEG subset is the interesting one: it's where the model has to predict
    DOWN against the default-UP grain, so good performance here is evidence of
    real signal. The POS subset is structurally hard to beat "always UP" on.
    """
    if "Article_Tone" not in df_h.columns:
        return []

    rows = []
    # "ALL" row for reference
    for tone in ["ALL", "POS", "NEG", "MIX"]:
        if tone == "ALL":
            sub = df_h.dropna(subset=["_pred_dir", "_act_dir"])
        else:
            sub = df_h[df_h["Article_Tone"] == tone].dropna(subset=["_pred_dir", "_act_dir"])
        n = len(sub)
        if n == 0:
            rows.append({
                "horizon": horizon, "tone": tone, "samples": 0,
                "actual_up_rate": np.nan, "pred_up_rate": np.nan,
                "accuracy": np.nan, "wf1": np.nan, "mcc": np.nan,
            })
            continue
        actual = sub["_act_dir"].values
        pred = sub["_pred_dir"].values
        rows.append({
            "horizon": horizon,
            "tone": tone,
            "samples": n,
            "actual_up_rate": float((actual == "UP").mean()),
            "pred_up_rate": float((pred == "UP").mean()),
            "accuracy": float((pred == actual).mean()),
            "wf1": _safe_wf1(actual.tolist(), pred.tolist()),
            "mcc": _safe_mcc(actual.tolist(), pred.tolist()),
        })
    return rows


def metrics_by_calibrated_threshold(df_h, horizon, thresholds):
    """For each P(UP) threshold, predict UP iff _p_up >= threshold; report
    accuracy / WF1 / MCC and the resulting pred_up_rate.

    With a balanced 50/50 ground truth, the threshold that yields
    pred_up_rate ≈ 0.5 typically maximizes MCC — it neutralizes the model's
    directional bias without re-running any LLM calls.
    """
    rows = []
    sub = df_h.dropna(subset=["_p_up", "_act_dir"])
    n = len(sub)
    actual = sub["_act_dir"].values if n > 0 else np.array([])
    actual_up_rate = float((actual == "UP").mean()) if n > 0 else np.nan

    for thr in thresholds:
        if n == 0:
            rows.append({
                "horizon": horizon, "threshold": thr, "samples": 0,
                "pred_up_rate": np.nan, "actual_up_rate": np.nan,
                "accuracy": np.nan, "wf1": np.nan, "mcc": np.nan,
            })
            continue
        pred = np.where(sub["_p_up"].values >= thr, "UP", "DOWN")
        rows.append({
            "horizon": horizon,
            "threshold": thr,
            "samples": n,
            "pred_up_rate": float((pred == "UP").mean()),
            "actual_up_rate": actual_up_rate,
            "accuracy": float((pred == actual).mean()),
            "wf1": _safe_wf1(actual.tolist(), pred.tolist()),
            "mcc": _safe_mcc(actual.tolist(), pred.tolist()),
        })
    return rows


def metrics_by_threshold(df_h, score_col, thresholds, denom):
    """For each threshold, keep rows with score_col >= threshold and report metrics.

    `denom` is the size of the full evaluable pool (used to compute coverage).
    Rows where score_col is NaN are excluded.
    """
    rows = []
    sub_all = df_h.dropna(subset=[score_col])
    for thr in thresholds:
        sub = sub_all[sub_all[score_col] >= thr]
        n = len(sub)
        coverage = (n / denom) if denom > 0 else np.nan
        if n == 0:
            rows.append({
                "threshold": thr, "samples": 0, "coverage": coverage,
                "accuracy": np.nan, "wf1": np.nan, "mcc": np.nan,
            })
            continue
        y_true = sub["_act_dir"].tolist()
        y_pred = sub["_pred_dir"].tolist()
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rows.append({
            "threshold": thr,
            "samples": n,
            "coverage": coverage,
            "accuracy": acc,
            "wf1": _safe_wf1(y_true, y_pred),
            "mcc": _safe_mcc(y_true, y_pred),
        })
    return rows


def _fmt(v, width=12, decimals=4):
    if pd.isna(v):
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{decimals}f}"


def print_threshold_table(title, rows_df):
    print(f"\n📈 {title}")
    print(f"{'Horizon':<10}{'Thr':>6}{'Samples':>10}{'Coverage':>12}{'Accuracy':>12}{'WF1':>12}{'MCC':>12}")
    for _, r in rows_df.iterrows():
        cov = _fmt(r["coverage"])
        print(f"{r['horizon']:<10}{r['threshold']:>6.2f}{int(r['samples']):>10}"
              f"{cov}{_fmt(r['accuracy'])}{_fmt(r['wf1'])}{_fmt(r['mcc'])}")


def main():
    df = load_predictions_from_parquets()
    print(f"Loaded {len(df)} rows with predictions")

    horizons = find_horizons(df.columns)
    if not horizons:
        raise RuntimeError("No Pred_Direction_<h> columns found in evaluation data")

    print("Detected horizons:", horizons)
    has_relevance = "Pred_Relevance" in df.columns
    if not has_relevance:
        print("⚠️ Pred_Relevance column not found — skipping relevance threshold sweep")

    all_tickers = df["Ticker"].dropna().unique()

    # Column presence isn't enough: _row_to_eval_dict creates the column even
    # when the parquet lacks the underlying actual_majority_direction_{h} field
    # (every value becomes None). Require at least one non-null value before
    # running the day-count-majority section, otherwise we'd print a table of
    # all zeros / n/a.
    has_majority = any(
        f"Actual_Majority_Direction_{h}" in df.columns
        and df[f"Actual_Majority_Direction_{h}"].notna().any()
        for h in horizons
    )
    if not has_majority:
        print("⚠️ Actual_Majority_Direction_* columns are not populated — "
              "run backfill_majority_direction.py (update its PRED_DIR to "
              f"{PRED_DIR!r} first) to backfill from price data. "
              "Skipping majority-direction metrics.")

    all_summaries = []
    overall_rows = []
    overall_majority_rows = []
    confidence_rows = []
    relevance_rows = []
    magnitude_rows = []
    baseline_rows = []
    baseline_majority_rows = []
    calibration_rows = []
    tone_rows = []

    for h in horizons:
        print(f"\n🔍 Computing metrics for horizon: {h}")
        df_h = prepare_horizon_df(df, h)
        skipped = count_skipped(df, h)
        pre_filter_n = len(df_h)

        if PER_HORIZON_FILTER:
            df_h = apply_per_horizon_filter(df_h, h)
            n_up = int((df_h["_act_dir"] == "UP").sum())
            n_down = int((df_h["_act_dir"] == "DOWN").sum())
            print(f"  Valid (UP/DOWN) predictions: {pre_filter_n}, "
                  f"SKIPPED (gated): {skipped}")
            if FOUR_WAY_TONE_FILTER and "Article_Tone" in df_h.columns:
                n_pos = int((df_h["Article_Tone"] == "POS").sum())
                n_neg = int((df_h["Article_Tone"] == "NEG").sum())
                print(f"  4-way balanced filter (tone × outcome) "
                      f"→ {len(df_h)} rows (UP={n_up}, DOWN={n_down}, "
                      f"POS={n_pos}, NEG={n_neg})")
            else:
                print(f"  Per-horizon top-{PER_HORIZON_MAX_ARTICLES} balanced filter "
                      f"→ {len(df_h)} rows (UP={n_up}, DOWN={n_down})")
        else:
            print(f"  Valid (UP/DOWN) predictions: {pre_filter_n}, "
                  f"SKIPPED (gated): {skipped}")

        total = len(df_h)

        # Per-ticker accuracy
        summ = compute_accuracy_by_ticker(df_h, h, all_tickers)
        all_summaries.append(summ)

        out_path = os.path.join(OUT_DIR, f"accuracy_by_ticker_{h}.csv")
        summ.to_csv(out_path, index=False)
        print("  ✅ wrote:", out_path)

        # Overall — return-based ground truth
        overall_rows.append(compute_overall(df_h, h, skipped))

        # Overall — majority-direction ground truth (only if backfilled)
        if has_majority:
            maj = compute_overall(df_h, h, skipped, gt_col="_act_maj_dir")
            if maj is not None:
                overall_majority_rows.append(maj)

        # Confidence threshold sweep
        for r in metrics_by_threshold(df_h, "_pred_conf", CONFIDENCE_THRESHOLDS, total):
            r["horizon"] = h
            confidence_rows.append(r)

        # Relevance threshold sweep
        if has_relevance and df_h["_pred_rel"].notna().any():
            for r in metrics_by_threshold(df_h, "_pred_rel", RELEVANCE_THRESHOLDS, total):
                r["horizon"] = h
                relevance_rows.append(r)

        # Magnitude threshold sweep (filter out near-zero return labels)
        if df_h["_act_return"].notna().any():
            magnitude_rows.extend(metrics_by_magnitude_threshold(df_h, h, MAGNITUDE_THRESHOLDS))

        # Calibrated decision-threshold sweep — finds the cutoff on P(UP) that
        # corrects the model's directional bias on a 50/50 ground truth.
        if df_h["_p_up"].notna().any():
            calibration_rows.extend(
                metrics_by_calibrated_threshold(df_h, h, CALIBRATION_THRESHOLDS)
            )

        # Tone-stratified metrics (POS/NEG/MIX articles based on factor sentiment)
        if "Article_Tone" in df_h.columns and df_h["Article_Tone"].notna().any():
            tone_rows.extend(metrics_by_article_tone(df_h, h))

        # Baselines + class distribution — return-based
        baseline_rows.extend(compute_baselines(df_h, h))

        # Baselines + class distribution — majority-direction
        if has_majority:
            baseline_majority_rows.extend(
                compute_baselines(df_h, h, gt_col="_act_maj_dir")
            )

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "accuracy_by_ticker_all_horizons.csv")
    combined.to_csv(combined_path, index=False)
    print("\n✅ wrote:", combined_path)

    # ----------------------------------------------------------------
    # Overall metrics across all tickers, per horizon + grand total
    # ----------------------------------------------------------------
    print("\n📊 Overall metrics across all tickers (direction):")
    header = (f"{'Horizon':<10}{'Samples':>10}{'Skipped':>10}{'Accuracy':>12}"
              f"{'WF1':>12}{'MCC':>12}{'MCC 95% CI':>22}")
    print(header)
    print("-" * len(header))

    total_correct = 0
    total_samples = 0
    for m in overall_rows:
        if m["samples"] > 0 and not pd.isna(m["accuracy"]):
            total_correct += m["accuracy"] * m["samples"]
            total_samples += m["samples"]

        lo, hi = m.get("mcc_ci_low"), m.get("mcc_ci_high")
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) and pd.notna(hi) else "n/a"
        print(f"{m['horizon']:<10}{m['samples']:>10}{m['skipped']:>10}"
              f"{_fmt(m['accuracy'])}{_fmt(m['wf1'])}{_fmt(m['mcc'])}"
              f"{ci_str:>22}")

    total_acc = (total_correct / total_samples) if total_samples > 0 else float("nan")
    print("-" * len(header))
    if total_samples > 0:
        print(f"{'TOTAL':<10}{total_samples:>10}{'':>10}{total_acc:>12.4f}"
              f"{'':>12}{'':>12}")
    else:
        print(f"{'TOTAL':<10}{0:>10}{'':>10}{'n/a':>12}{'':>12}{'':>12}")

    overall_df = pd.DataFrame(overall_rows)
    overall_path = os.path.join(OUT_DIR, "overall_metrics_by_horizon.csv")
    overall_df.to_csv(overall_path, index=False)
    print("\n✅ wrote:", overall_path)

    # ----------------------------------------------------------------
    # Confidence threshold sweep
    # ----------------------------------------------------------------
    if confidence_rows:
        conf_df = pd.DataFrame(confidence_rows)[
            ["horizon", "threshold", "samples", "coverage", "accuracy", "wf1", "mcc"]
        ]
        conf_path = os.path.join(OUT_DIR, "metrics_by_confidence_threshold.csv")
        conf_df.to_csv(conf_path, index=False)
        print_threshold_table("Accuracy by confidence threshold (per horizon)", conf_df)
        print("\n✅ wrote:", conf_path)

    # ----------------------------------------------------------------
    # Relevance threshold sweep
    # ----------------------------------------------------------------
    if relevance_rows:
        rel_df = pd.DataFrame(relevance_rows)[
            ["horizon", "threshold", "samples", "coverage", "accuracy", "wf1", "mcc"]
        ]
        rel_path = os.path.join(OUT_DIR, "metrics_by_relevance_threshold.csv")
        rel_df.to_csv(rel_path, index=False)
        print_threshold_table("Accuracy by relevance threshold (per horizon)", rel_df)
        print("\n✅ wrote:", rel_path)

    # ----------------------------------------------------------------
    # Magnitude threshold sweep (|Actual_Return| filter)
    # ----------------------------------------------------------------
    if magnitude_rows:
        mag_df = pd.DataFrame(magnitude_rows)[
            ["horizon", "threshold", "samples", "coverage", "accuracy", "wf1", "mcc"]
        ]
        mag_path = os.path.join(OUT_DIR, "metrics_by_magnitude_threshold.csv")
        mag_df.to_csv(mag_path, index=False)
        print_threshold_table(
            "Accuracy by |Actual_Return| threshold — drops near-zero returns",
            mag_df,
        )
        print("\n✅ wrote:", mag_path)

    # ----------------------------------------------------------------
    # Calibrated decision-threshold sweep
    # ----------------------------------------------------------------
    if calibration_rows:
        cal_df = pd.DataFrame(calibration_rows)[
            ["horizon", "threshold", "samples", "pred_up_rate", "actual_up_rate",
             "accuracy", "wf1", "mcc"]
        ]
        cal_path = os.path.join(OUT_DIR, "metrics_by_calibrated_threshold.csv")
        cal_df.to_csv(cal_path, index=False)

        print("\n📈 Calibrated decision-threshold sweep — predict UP iff P(UP) >= t")
        print("   (P(UP) = vote_up/K from the K=5 sampling; threshold=0.5 == majority vote)")
        print(f"{'Horizon':<8}{'Thr':>6}{'Samples':>9}{'PredUP%':>10}{'ActUP%':>9}"
              f"{'Accuracy':>11}{'WF1':>10}{'MCC':>10}")
        for _, r in cal_df.iterrows():
            print(f"{r['horizon']:<8}{r['threshold']:>6.2f}{int(r['samples']):>9}"
                  f"{r['pred_up_rate']*100:>9.1f}%{r['actual_up_rate']*100:>8.1f}%"
                  f"{_fmt(r['accuracy'], width=11)}{_fmt(r['wf1'], width=10)}"
                  f"{_fmt(r['mcc'], width=10)}")
        print("\n✅ wrote:", cal_path)

        # Highlight the best threshold per horizon + the "balanced" threshold
        # (one whose pred_up_rate is closest to actual_up_rate, i.e. 0.5 under
        # a balanced eval set).
        print("\n📌 Best calibration per horizon (vs threshold=0.50 default):")
        print(f"{'Horizon':<8}{'BestThr':>8}{'BestMCC':>9}{'BestAcc':>9}"
              f"{'BalThr':>8}{'BalMCC':>9}{'BalAcc':>9}"
              f"{'BalPredUP%':>12}{'Δ vs t=.5':>12}")
        for h in horizons:
            sub = cal_df[cal_df["horizon"] == h]
            if sub.empty or sub["mcc"].isna().all():
                continue
            best = sub.loc[sub["mcc"].idxmax()]
            sub2 = sub.assign(_gap=(sub["pred_up_rate"] - sub["actual_up_rate"]).abs())
            bal = sub2.loc[sub2["_gap"].idxmin()]
            default = sub[sub["threshold"] == 0.5]
            default_mcc = float(default["mcc"].iloc[0]) if not default.empty else np.nan
            delta_mcc = float(bal["mcc"]) - default_mcc if not np.isnan(default_mcc) else np.nan
            print(f"{h:<8}{best['threshold']:>8.2f}{best['mcc']:>9.4f}{best['accuracy']:>9.4f}"
                  f"{bal['threshold']:>8.2f}{bal['mcc']:>9.4f}{bal['accuracy']:>9.4f}"
                  f"{bal['pred_up_rate']*100:>11.1f}%"
                  f"{delta_mcc:>+12.4f}")

    # ----------------------------------------------------------------
    # Tone-stratified metrics (POS / NEG / MIX article subsets)
    # ----------------------------------------------------------------
    if tone_rows:
        tone_df = pd.DataFrame(tone_rows)[
            ["horizon", "tone", "samples", "actual_up_rate", "pred_up_rate",
             "accuracy", "wf1", "mcc"]
        ]
        tone_path = os.path.join(OUT_DIR, "metrics_by_article_tone.csv")
        tone_df.to_csv(tone_path, index=False)

        print("\n📈 Metrics stratified by article tone (POS = more positive factors than negative)")
        print("   POS subset: structurally hard — most articles default toward UP")
        print("   NEG subset: where the model has to go against the default — real signal lives here")
        print(f"{'Horizon':<8}{'Tone':>6}{'Samples':>9}{'ActUP%':>9}{'PredUP%':>9}"
              f"{'Accuracy':>11}{'WF1':>10}{'MCC':>10}")
        for _, r in tone_df.iterrows():
            print(f"{r['horizon']:<8}{r['tone']:>6}{int(r['samples']):>9}"
                  f"{r['actual_up_rate']*100:>8.1f}%{r['pred_up_rate']*100:>8.1f}%"
                  f"{_fmt(r['accuracy'], width=11)}{_fmt(r['wf1'], width=10)}"
                  f"{_fmt(r['mcc'], width=10)}")
        print("\n✅ wrote:", tone_path)

    # ----------------------------------------------------------------
    # Baselines + class distribution
    # ----------------------------------------------------------------
    if baseline_rows:
        bl_df = pd.DataFrame(baseline_rows)
        bl_path = os.path.join(OUT_DIR, "baselines_by_horizon.csv")
        bl_df.to_csv(bl_path, index=False)

        print("\n📊 Baselines vs pipeline (does the LLM add value?):")
        print(f"{'Horizon':<10}{'Baseline':<26}{'Samples':>10}{'Accuracy':>12}{'MCC':>12}")
        for _, r in bl_df.iterrows():
            print(f"{r['horizon']:<10}{str(r['baseline']):<26}{int(r['samples']):>10}"
                  f"{_fmt(r['accuracy'])}{_fmt(r['mcc'])}")
        print("\n  Note: rows with baseline='_class_dist' show actual_up_frac and pred_up_frac.")
        print("✅ wrote:", bl_path)

    # ================================================================
    # Same metrics, scored against day-count majority direction
    # (the alternate ground truth — UP = more UP days than DOWN days)
    # ================================================================
    if overall_majority_rows:
        print("\n" + "=" * 78)
        print("📊 Overall metrics — vs DAY-COUNT MAJORITY direction:")
        print("=" * 78)
        header = f"{'Horizon':<10}{'Samples':>10}{'Skipped':>10}{'Accuracy':>12}{'WF1':>12}{'MCC':>12}"
        print(header)
        print("-" * len(header))
        for m in overall_majority_rows:
            print(f"{m['horizon']:<10}{m['samples']:>10}{m['skipped']:>10}"
                  f"{_fmt(m['accuracy'])}{_fmt(m['wf1'])}{_fmt(m['mcc'])}")

        overall_majority_df = pd.DataFrame(overall_majority_rows)
        overall_majority_path = os.path.join(OUT_DIR, "overall_metrics_by_horizon_majority.csv")
        overall_majority_df.to_csv(overall_majority_path, index=False)
        print("\n✅ wrote:", overall_majority_path)

    if baseline_majority_rows:
        bl_maj_df = pd.DataFrame(baseline_majority_rows)
        bl_maj_path = os.path.join(OUT_DIR, "baselines_by_horizon_majority.csv")
        bl_maj_df.to_csv(bl_maj_path, index=False)

        print("\n📊 Baselines vs pipeline — DAY-COUNT MAJORITY ground truth:")
        print(f"{'Horizon':<10}{'Baseline':<26}{'Samples':>10}{'Accuracy':>12}{'MCC':>12}")
        for _, r in bl_maj_df.iterrows():
            print(f"{r['horizon']:<10}{str(r['baseline']):<26}{int(r['samples']):>10}"
                  f"{_fmt(r['accuracy'])}{_fmt(r['mcc'])}")
        print("✅ wrote:", bl_maj_path)


if __name__ == "__main__":
    main()
