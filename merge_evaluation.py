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

PRED_DIR = "data/prediction_batches_qwen"   # source of truth (live parquets)
EVAL_DIR = "data/evaluation_results_qwen"   # legacy CSV fallback
OUT_DIR = "data/evaluation_summary_qwen"

os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = ["1d", "3d", "5d", "10d", "21d"]

CONFIDENCE_THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RELEVANCE_THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# |Actual_Return| filter — drops near-zero return labels which are essentially noise.
MAGNITUDE_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

VALID_DIRECTIONS = {"UP", "DOWN"}

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
    df_h = df[cols].copy()

    df_h = df_h.dropna(subset=[pred_dir_col, act_dir_col])

    df_h["_pred_dir"] = df_h[pred_dir_col].astype(str).str.strip().str.upper()
    df_h["_act_dir"] = df_h[act_dir_col].astype(str).str.strip().str.upper()

    # Filtering to UP/DOWN automatically drops SKIPPED rows
    df_h = df_h[df_h["_act_dir"].isin(VALID_DIRECTIONS)]
    df_h = df_h[df_h["_pred_dir"].isin(VALID_DIRECTIONS)]

    df_h["_pred_conf"] = pd.to_numeric(df_h[pred_conf_col], errors="coerce")
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

    wf1_mcc = (
        df_h.groupby("Ticker")
            .apply(lambda g: pd.Series({
                "direction_wf1": _safe_wf1(g["_act_dir"].tolist(), g["_pred_dir"].tolist()),
                "direction_mcc": _safe_mcc(g["_act_dir"].tolist(), g["_pred_dir"].tolist()),
            }))
            .reset_index()
    )

    summary = samples.merge(accuracy, on="Ticker", how="left").merge(wf1_mcc, on="Ticker", how="left")
    summary["horizon"] = horizon
    return summary[out_cols]


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
    return {
        "horizon": horizon,
        "samples": n,
        "skipped": skipped_count,
        "accuracy": acc,
        "wf1": _safe_wf1(y_true, y_pred),
        "mcc": _safe_mcc(y_true, y_pred),
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

    has_majority = any(f"Actual_Majority_Direction_{h}" in df.columns for h in horizons)
    if not has_majority:
        print("⚠️ Actual_Majority_Direction_* columns not found — "
              "run backfill_majority_direction.py to populate them. "
              "Skipping majority-direction metrics.")

    all_summaries = []
    overall_rows = []
    overall_majority_rows = []
    confidence_rows = []
    relevance_rows = []
    magnitude_rows = []
    baseline_rows = []
    baseline_majority_rows = []

    for h in horizons:
        print(f"\n🔍 Computing metrics for horizon: {h}")
        df_h = prepare_horizon_df(df, h)
        skipped = count_skipped(df, h)
        total = len(df_h)
        print(f"  Valid (UP/DOWN) predictions: {total}, SKIPPED (gated): {skipped}")

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
    header = f"{'Horizon':<10}{'Samples':>10}{'Skipped':>10}{'Accuracy':>12}{'WF1':>12}{'MCC':>12}"
    print(header)
    print("-" * len(header))

    total_correct = 0
    total_samples = 0
    for m in overall_rows:
        if m["samples"] > 0 and not pd.isna(m["accuracy"]):
            total_correct += m["accuracy"] * m["samples"]
            total_samples += m["samples"]

        print(f"{m['horizon']:<10}{m['samples']:>10}{m['skipped']:>10}"
              f"{_fmt(m['accuracy'])}{_fmt(m['wf1'])}{_fmt(m['mcc'])}")

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
