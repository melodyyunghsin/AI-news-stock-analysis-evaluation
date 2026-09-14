"""Multiple-comparison correction on the per-(ticker x horizon) MCC results.

SUPPLEMENTARY POST-HOC ANALYSIS — this is NOT part of the main prediction or
scoring pipeline (merge_evaluation.py computes only per-cell bootstrap 95% CIs).
It runs on the committed output of that pipeline to answer a single question a
reviewer will ask: "you tested 35 (ticker x horizon) cells — did you correct
for multiple comparisons?"

Method
------
For a 2x2 confusion matrix the Pearson chi-square statistic equals N * MCC^2
(df = 1), which is a standard test of association (i.e. of MCC != 0). We turn
each cell's (N, MCC) into a p-value that way, then apply:

  * Bonferroni      — strict: reject only if p < alpha / m  (controls the
                      family-wise error rate; chance of ANY false positive).
  * Benjamini-Hochberg FDR — lenient: controls the expected PROPORTION of
                      false positives among the cells declared significant.

For context we also report the per-cell bootstrap-CI verdict already in the
CSV (significant if the 95% CI excludes 0) — that is the test the in-extension
reliability badge uses. The two methods can disagree on borderline cells
(e.g. TSM@10d), but both agree on the headline: nothing survives correction.

Input : data/balanced_focused_evaluation_summary_gemini_k5/accuracy_by_ticker_all_horizons.csv
Output: data/balanced_focused_evaluation_summary_gemini_k5/multiple_comparison_results.csv

Run:  python multiple_comparison_correction.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONFIG
# ============================================================

SUMMARY_DIR = "data/balanced_focused_evaluation_summary_gemini_k5"
IN_CSV = os.path.join(SUMMARY_DIR, "accuracy_by_ticker_all_horizons.csv")
OUT_CSV = os.path.join(SUMMARY_DIR, "multiple_comparison_results.csv")
ALPHA = 0.05


def main():
    df = pd.read_csv(IN_CSV)
    # Keep only rows with a usable MCC + sample count
    df = df.dropna(subset=["direction_mcc", "samples"]).copy()
    df["samples"] = df["samples"].astype(int)

    m = len(df)

    # ---- per-cell chi-square p-value for MCC != 0 ----
    df["chi2"] = df["samples"] * df["direction_mcc"] ** 2
    df["p_value"] = stats.chi2.sf(df["chi2"], df=1)

    # ---- uncorrected verdicts ----
    df["sig_p_uncorrected"] = df["p_value"] < ALPHA            # chi-square test
    df["sig_ci_uncorrected"] = (df["direction_mcc_ci_low"] > 0) | (
        df["direction_mcc_ci_high"] < 0)                       # bootstrap-CI (badge)

    # ---- Bonferroni ----
    bonf_threshold = ALPHA / m
    df["sig_bonferroni"] = df["p_value"] < bonf_threshold

    # ---- Benjamini-Hochberg FDR ----
    df = df.sort_values("p_value").reset_index(drop=True)
    rank = np.arange(1, m + 1)
    bh_threshold = rank / m * ALPHA
    passes = df["p_value"].values <= bh_threshold
    # BH: significant up to the largest rank that passes
    k = np.max(np.where(passes)[0]) + 1 if passes.any() else 0
    df["sig_fdr"] = False
    if k > 0:
        df.loc[: k - 1, "sig_fdr"] = True

    # ---- write ----
    cols = ["Ticker", "horizon", "samples", "direction_mcc",
            "direction_mcc_ci_low", "direction_mcc_ci_high",
            "chi2", "p_value", "sig_ci_uncorrected", "sig_p_uncorrected",
            "sig_bonferroni", "sig_fdr"]
    df[cols].to_csv(OUT_CSV, index=False)

    # ---- summary ----
    exp_fp = m * ALPHA
    n_ci = int(df["sig_ci_uncorrected"].sum())
    n_p = int(df["sig_p_uncorrected"].sum())
    n_bonf = int(df["sig_bonferroni"].sum())
    n_fdr = int(df["sig_fdr"].sum())
    p_at_least = stats.binom.sf(n_p - 1, m, ALPHA) if n_p > 0 else 1.0

    print("=" * 66)
    print("Multiple-comparison correction on per-(ticker x horizon) MCC")
    print("=" * 66)
    print(f"Cells tested (m)                     : {m}")
    print(f"Expected false positives @ a={ALPHA}    : {exp_fp:.2f}")
    print(f"Significant — bootstrap CI (badge)   : {n_ci}")
    print(f"Significant — chi-square p<{ALPHA}      : {n_p}  "
          f"(P(>= {n_p} by chance) = {p_at_least:.3f})")
    print(f"Survive Bonferroni (p < {bonf_threshold:.5f})    : {n_bonf}")
    print(f"Survive Benjamini-Hochberg FDR       : {n_fdr}")
    print("-" * 66)
    print("Smallest p-values:")
    show = df.head(6)[["Ticker", "horizon", "samples", "direction_mcc",
                       "p_value", "sig_bonferroni", "sig_fdr"]]
    print(show.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    print("=" * 66)
    print(f"Wrote {OUT_CSV}")
    if n_bonf == 0 and n_fdr == 0:
        print("Conclusion: 0 cells survive correction — consistent with "
              "overall MCC ~ 0 (near chance).")


if __name__ == "__main__":
    main()
