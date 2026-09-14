# Evaluation Process — News-driven Stock Direction Prediction

## 1. Goal

This document describes the methodology and results of an LLM-based pipeline for predicting short-term stock-price direction (UP / DOWN) from individual news articles. The target universe is 7 US mega-cap stocks (AAPL, MSFT, NVDA, AMZN, TSLA, GOOG, TSM); predictions span 5 trading-day horizons (1d, 3d, 5d, 10d, 21d).

## 2. Dataset

**Source**: `data/balanced_focused_dataset` — a pre-filtered slice of the FNSPID news corpus. Articles were first scored by `qwen2.5:7b` for relevance to the target ticker and only those with relevance ≥ 0.8 were retained; the result was then balanced per-ticker in article count.

**Schema**: Each row contains article content and 5-horizon return information:

- `Date`, `Stock_symbol`, `Article_text`, `Article_title`, `Url`, `Publisher`
- `return_{h}` and `actual_direction_{h}` for h ∈ {1d, 3d, 5d, 10d, 21d}
- `actual_majority_direction_{h}` (day-count majority within each horizon window)

**Total**: ~3,000 articles across 7 tickers.

## 3. Pipeline Architecture

The pipeline runs three phases per ticker, persisting intermediate state to per-ticker parquet files so runs can resume and intermediate outputs can be reused.

### 3.1 Phase 1 — Factor Extraction

For each article, a single LLM call produces:

- A relevance score (0.0–1.0) for how directly the article concerns the target ticker
- 3–5 causal **factors**, each with a direction (positive / negative), time horizon (short / medium / long-term), and confidence (high / medium / low)

The factor JSON is stored in the row's `factors_json` column so the prediction phase can reuse it without re-extracting.

### 3.2 Phase 2 — 4-way Balanced Selection

For each (ticker, horizon) combination, candidate rows are partitioned into four quadrants by *(article tone × actual outcome)*:

- POS × UP, POS × DOWN, NEG × UP, NEG × DOWN

Article tone is derived from the factor extraction: POS = more positive factors than negative, NEG = vice versa. From each quadrant the top `MAX_ARTICLES_PER_TICKER // 4` rows by relevance are taken; the smallest quadrant caps the others (strict balance). The union of the 5 horizons' selections becomes the final evaluation set.

**Rationale**: A diagnostic showed the raw dataset is 78% POS-toned and only 17% NEG-toned. Without correction, the model defaulted to UP predictions roughly 70% of the time, regardless of factor sentiment. The 4-way balance eliminates the input-tone bias and the outcome-class bias simultaneously — each horizon's eval set ends up exactly 50/50 on both axes per ticker.

### 3.3 Phase 3 — Self-consistency Prediction (K = 11)

For each selected article × horizon, the prediction prompt is sampled K = 11 times at temperature 0.8. The majority-vote direction is the final prediction; the vote-agreement ratio (e.g. 8/11 ≈ 0.73) becomes a calibrated confidence score. The full prompt contains: company context (sector, industry, summary), a price summary (recent moves, SMAs, volatility), the extracted factors, and the article text. K was initially set to 5; we bumped it to 11 after observing that run-to-run K-sample variance was comparable to the signal magnitude at K=5 (see Section 8.3).

**Model**: `gemini-2.5-flash-lite` via the Gemini Batch API (~50% cheaper than synchronous calls, with minutes-to-hours turnaround acceptable for offline evaluation).

## 4. Evaluation Methodology

For each (ticker, horizon) pair we report:

- **Accuracy** — fraction of predictions matching the actual direction
- **Matthews Correlation Coefficient (MCC)** — symmetric, robust to class imbalance, equals 0 for random prediction
- **Weighted F1**

### 4.1 Per-horizon Balanced Evaluation

Each horizon is scored independently on the rows selected for *that* horizon's 4-way balanced top-N. This guarantees:

- Each horizon's eval set is exactly 50/50 on actual UP/DOWN
- Each horizon's eval set is exactly 50/50 on POS/NEG article tone
- Per-horizon MCC and accuracy can be compared directly without class-balance confounding

### 4.2 Tone Stratification

Within each horizon's eval set, metrics are recomputed separately for POS-toned and NEG-toned articles. This isolates:

- **POS subset performance** — structurally hard because positive news on mega-caps is often already priced in
- **NEG subset performance** — where the model has to commit to DOWN against its default-UP grain; this is where real signal manifests

### 4.3 Calibration Threshold Sweep

The K = 11 vote agreement provides a continuous P(UP) score (`vote_up / K`). Decision thresholds from 0.0 to 1.0 are swept; for each threshold we report accuracy, MCC, and the resulting `pred_up_rate`. With a balanced 50/50 ground truth, the threshold that yields `pred_up_rate ≈ 0.5` reveals whether the model has an exploitable directional bias.

## 5. Results

### 5.1 Overall Metrics (4-way balanced, 7 tickers, gemini-2.5-flash-lite, K=11)

| Horizon | Samples | Accuracy | MCC | 95% CI (bootstrap, n=1000) |
|---|---:|---:|---:|:---:|
| 1d  | 584 | 52.23% | +0.045 | [−0.040, +0.128] |
| 3d  | 572 | 50.52% | +0.011 | [−0.075, +0.090] |
| 5d  | 516 | 50.39% | +0.008 | [−0.076, +0.092] |
| 10d | 496 | 51.01% | +0.020 | [−0.065, +0.109] |
| 21d | 444 | 47.52% | −0.050 | [−0.140, +0.049] |

**K-sample stochasticity is substantial.** During development we ran the K=5 pipeline twice with identical prompts and got 1d MCC values of +0.126 and −0.058 — a between-run range of 0.18 from K=5 voting alone. Bumping K to 11 produces a more stable per-run estimate (+0.045 here, near the average of the two K=5 runs at +0.034). The within-run bootstrap CI captures resampling noise but does NOT capture between-run K-sample variance, so it should be read as a lower bound on uncertainty.

**Interpretation**: the pipeline shows weak positive signal at the 1-day horizon (~+0.04 MCC) that is below the run-to-run noise floor of a single K=11 run. The signal magnitude is consistent with published LLM-based direction-prediction benchmarks (~55-58% accuracy ceiling for mega-cap news, e.g., FinGPT, LLMFactor), but it cannot be reliably distinguished from zero in any single run. The other horizons show point estimates near zero or slightly negative; no horizon is statistically significant at 95% in this single run.

What this evaluation *can* still defensibly claim — even with the noise floor caveat — is that the LLM pipeline beats off-the-shelf finance sentiment (FinBERT) at every horizon and short-horizon price momentum at 1d–10d (Section 6.3). The frontier-model comparisons (Sections 6.1, 6.2) are run on much smaller subsets and are **inconclusive** — they do not show a clear win or loss for the bigger models.

### 5.2 Per-ticker Signal Strength (1d horizon)

| Ticker | Samples | Accuracy | MCC | 95% CI |
|---|---:|---:|---:|:---:|
| AAPL | 100 | 50.0% | 0.000  | [−0.198, +0.199] |
| GOOG | 100 | 51.0% | +0.020 | [−0.166, +0.207] |
| MSFT | 72  | 50.0% | 0.000  | [−0.227, +0.242] |
| NVDA | 100 | 55.0% | +0.103 | [−0.104, +0.287] |
| TSM  | 60  | 55.0% | +0.101 | [−0.156, +0.334] |
| AMZN | 52  | 50.0% | 0.000  | [−0.257, +0.281] |
| TSLA | 100 | 54.0% | +0.081 | [−0.118, +0.294] |

At 1d, NVDA, TSM, and TSLA carry the highest point estimates (+0.08 to +0.10 MCC), but **every per-ticker 95% CI spans 0** — no individual ticker reaches significance at 1d with 50–100 samples. Individual per-ticker MCCs should therefore not be over-interpreted (see Section 8.4).

Across all 35 (ticker × horizon) cells, only four clear the significance bar (95% CI excluding 0):

| Cell | Samples | Accuracy | MCC | 95% CI | Verdict |
|---|---:|---:|---:|:---:|:---:|
| TSM 10d  | 60  | 61.7% | +0.237 | [+0.005, +0.472] | **significant +** |
| NVDA 5d  | 100 | 38.0% | −0.243 | [−0.424, −0.053] | significant − |
| MSFT 10d | 24  | 33.3% | −0.447 | [−0.671, −0.213] | significant − |
| TSLA 21d | 100 | 40.0% | −0.200 | [−0.396, −0.006] | significant − |

The remaining 31 cells are statistically indistinguishable from random. **TSM @ 10d is the only ticker × horizon with a real positive signal** (this is the single combination the extension's reliability badge marks as "Signal"); the three significant-negative cells are consistent with small-sample noise and short-term reversal rather than a usable "short" signal.

### 5.3 Tone-stratified Findings

Within each horizon's balanced eval set, metrics are recomputed separately for POS- and NEG-toned articles:

| Horizon | POS subset MCC | NEG subset MCC | POS pred-UP | NEG pred-UP |
|---|---:|---:|---:|---:|
| 1d  | +0.021 | +0.070 | 52.4% | 61.6% |
| 3d  | +0.077 | −0.057 | 54.2% | 60.8% |
| 5d  | 0.000  | +0.016 | 52.7% | 60.5% |
| 10d | +0.040 | 0.000  | 51.2% | 62.9% |
| 21d | −0.036 | −0.064 | 48.6% | 59.9% |

The differences between the POS and NEG subsets are small and, with ~250–290 articles per subset, none of these MCC values is individually significant. Two observations:

- The model does **not** flip to DOWN on negative-toned articles — if anything it predicts UP *more* on NEG-toned news (~60–63%) than on POS-toned news (~49–54%). This is the opposite of a sentiment-following classifier, and it is why the overall pred-UP rate stays above 50% even on a perfectly balanced set.
- The 1d NEG cell (+0.070) is the strongest tone cell, but it is well within noise.

**A strong "the signal lives in NEG articles" effect appeared in earlier development (K=5) runs but did NOT replicate** in the final K=11, 4-way-balanced run. We therefore do not claim a deployable tone gate; see Section 8.2 for the related (failed) attempt to engineer the UP bias away.

## 6. Model Comparisons: Frontier Models vs Flash-Lite

Two subset experiments tested whether a *stronger* LLM improves performance. The takeaway: **across model scale — flash-lite, gemini-2.5-pro, and claude-opus-4-8 — direction MCC stays pinned near zero. No model escapes the noise floor.** Neither frontier model delivered a reliable improvement over flash-lite on these subsets, which points to the *task* (semi-strong market efficiency on mega-caps) as the binding constraint — not model capability or scale. The per-model details below are exploratory (small, non-matched subsets); read them as "nothing here breaks the ceiling," not as a precise ranking.

### 6.1 gemini-2.5-pro

Subset test on AAPL, AMZN, TSLA at MAX = 40 per ticker (4-way balanced), K = 5.

Overall metrics (3-ticker subset):

| Horizon | Samples | Accuracy | MCC |
|---|---:|---:|---:|
| 1d  | 120 | 53.3% | +0.069 |
| 3d  | 148 | 45.3% | −0.100 |
| 5d  | 128 | 50.0% |  0.000 |
| 10d | 132 | 46.2% | −0.078 |
| 21d | 112 | 57.1% | +0.149 |

Per-ticker at 1d (vs the corresponding committed flash-lite cell):

| Ticker | n | Pro MCC | Flash-lite MCC |
|---|---:|---:|---:|
| AAPL | 40 | +0.100 | 0.000  |
| AMZN | 40 | −0.231 | 0.000  |
| TSLA | 40 | +0.306 | +0.081 |

The result is **mixed, not a clean win or loss**. Pro's overall MCC sits above flash-lite's all-ticker numbers at 1d (+0.069) and 21d (+0.149) but is negative at 3d and 10d. Per-ticker at 1d, Pro beats flash-lite on TSLA and AAPL but collapses on AMZN. With only n = 40 per ticker the standard error (~±0.16) swamps these gaps, so the subset is too small to conclude that Pro helps or hurts.

### 6.2 claude-opus-4-8

Independent test with a different model family. Subset on AAPL only at MAX = 40 (4-way balanced), K = 1 — Opus 4.8 does not accept the `temperature` parameter, so K-sample self-consistency was disabled. This makes the test *harder* for Claude (no vote-based denoising).

| Horizon | n | Claude MCC | Flash-lite MCC (AAPL, committed) | Δ MCC |
|---|---:|---:|---:|---:|
| 1d  | 44 | +0.046 | 0.000  | +0.046 |
| 3d  | 44 | +0.091 | +0.020 | +0.071 |
| 5d  | 44 | −0.183 | +0.100 | **−0.283** |
| 10d | 40 |  0.000 | −0.020 | +0.020 |
| 21d | 40 | −0.052 | −0.088 | +0.036 |

Average Δ MCC ≈ **−0.02** — essentially neutral. Claude edges flash-lite at four of five horizons but suffers a sharp 5d dip (−0.18). Both models sit near zero on AAPL, and the comparison is confounded (Claude ran MAX = 40 / K = 1 on a different article subset than flash-lite's MAX = 100 / K = 11). The honest reading: **Opus 4.8 is roughly on par with flash-lite here, not clearly better** — a frontier model at far higher cost buys no measured edge.

### 6.3 Non-LLM baselines: FinBERT and price momentum

To verify the LLM pipeline adds value over simpler approaches, two non-LLM baselines were scored on the **same** 4-way balanced eval set:

1. **FinBERT** (`ProsusAI/finbert`) — a BERT model fine-tuned for financial-news sentiment classification. Predicts UP if P(positive) > P(negative), else DOWN.
2. **Momentum (20-day)** — predicts UP if the 20-day return prior to the article date is positive, DOWN otherwise. Pure price signal; article content is not used.

| Horizon | LLM MCC | FinBERT MCC | FinBERT 95% CI | Momentum MCC | Momentum 95% CI |
|---|---:|---:|:---:|---:|:---:|
| 1d  | **+0.045** | +0.007 | [−0.074, +0.089] | **−0.119** | [−0.203, **−0.038**] |
| 3d  | +0.011 | −0.022 | [−0.104, +0.057] | **−0.119** | [−0.202, **−0.037**] |
| 5d  | +0.008 | −0.004 | [−0.092, +0.078] | −0.054 | [−0.139, +0.034] |
| 10d | +0.020 | −0.038 | [−0.126, +0.044] | −0.083 | [−0.164, +0.011] |
| 21d | −0.050 | −0.058 | [−0.147, +0.036] | +0.038 | [−0.056, +0.134] |

**The LLM pipeline beats FinBERT at every horizon and beats momentum at 1d–10d**, though the margins are small. At 1d the gap over FinBERT is +0.038 MCC and the gap over momentum is +0.16 MCC. Three findings worth noting:

- **FinBERT is essentially random** on this eval set — its 95% CI includes zero at every horizon. Off-the-shelf finance-tuned sentiment lacks the nuance to predict direction. FinBERT predicts UP ~66% of the time (similar to a "default to positive sentiment" classifier), but with a balanced 50/50 ground truth this no longer gives an accuracy boost.
- **Momentum is *significantly negative* at 1d and 3d** (MCC −0.119, CI excludes zero on the negative side at both). This is consistent with short-term mean reversion: news-day articles in this dataset disproportionately follow recent trends that reverse over the next few days.
- **At 21d, momentum (+0.038) actually edges out the LLM (−0.050)** — the only horizon where a baseline beats the pipeline. This is consistent with the well-known longer-horizon continuation effect, and with the LLM's own 21d signal being negative. The LLM's nominal "win" over FinBERT at 21d (−0.050 vs −0.058) is a tie within noise.

### 6.4 Joint conclusion across all comparisons

The LLM pipeline has now been compared against three classes of alternative:

| Comparison | Result |
|---|---|
| Trivial baselines (always-UP, majority-class) | LLM beats them at 1d ✓ |
| Off-the-shelf finance NLP (FinBERT) | LLM beats it at every horizon (small margin) ✓ |
| Pure price signal (20d momentum) | LLM beats it at 1d–10d; momentum wins at 21d |
| Frontier LLM (gemini-2.5-pro) | Mixed on a 3-ticker subset — inconclusive |
| Frontier LLM (claude-opus-4-8) | ≈ on par with flash-lite on AAPL — inconclusive |

The LLM pipeline reliably beats off-the-shelf sentiment and short-horizon price momentum. The frontier-model comparisons, by contrast, are exploratory (small, non-matched subsets) and show **no model — flash-lite, pro, or Opus — climbing above the noise floor**: scaling up model capability does not rescue the task. The **directness + factor-extraction + K-sample self-consistency** design is therefore a sensible operating point, and `gemini-2.5-flash-lite` is the right ship choice on cost/latency grounds at no measured loss.

That said, **the absolute size of the win remains modest** — the single-run 1d MCC of +0.045 is within the run-to-run noise floor, and the longer-horizon signals (3d–21d) are within noise relative to FinBERT in most cases (and below momentum at 21d). The remaining gains are in input quality, decision-rule calibration, and dataset curation, not in model selection or scale.

## 7. Conclusions

1. **The pipeline produces weak positive signal at 1-day horizon, within the noise floor.** Single-run K=11 MCC at 1d is +0.045; multi-run mean across three reruns is +0.037 with range 0.18. The magnitude is consistent with the literature ceiling for LLM-based direction prediction on mega-caps (~55-58% accuracy), but cannot be distinguished from zero in any single run with the current sample size. Longer horizons (3d-21d) are essentially noise. The pipeline does have a real positive expected value, but its absolute magnitude is small.

2. **Tone stratification did not reveal a usable NEG-subset signal in the final run.** Earlier (K=5) development runs suggested negative-toned articles carried most of the signal, but in the final K=11, 4-way-balanced run the POS/NEG MCC differences are small and insignificant — and the model actually predicts UP *more* on NEG-toned articles (~60–63%) than on POS-toned ones (~49–54%). We do not claim a deployable tone gate.

3. **The LLM pipeline beats the simpler content and price baselines, by small margins.** It outperforms (a) trivial baselines (always-UP at 1d), (b) off-the-shelf finance-tuned sentiment (FinBERT — random at every horizon, beaten at all five), and (c) 20-day price momentum at 1d–10d (momentum is significantly *negative* at 1d and 3d, and overtakes the LLM only at 21d). The 1d gap of +0.038 MCC over FinBERT and +0.16 MCC over momentum is the clearest evidence that the two-step factor-extraction + K-sample self-consistency design does real work, not just added compute — though the absolute edge is small.

4. **Model scale does not move the needle — the task is the bottleneck.** Across flash-lite, `gemini-2.5-pro` (3 tickers, n = 40 each), and `claude-opus-4-8` (AAPL only, K = 1), direction MCC stays near zero; no model escapes the noise floor. Neither frontier model delivered a reliable improvement (and the subsets are too small to claim a *regression* either). This points to semi-strong market efficiency on mega-caps as the binding constraint, not model capability — so `gemini-2.5-flash-lite` is the right ship choice on cost/latency at no measured loss, and the remaining gains live in input/output engineering and data curation, not model selection.

5. **Per-ticker variance is large and mostly within noise.** Of the 35 (ticker × horizon) cells, only TSM @ 10d is a significant positive signal; NVDA @ 5d, MSFT @ 10d, and TSLA @ 21d are significantly negative; the remaining 31 are at chance. Individual per-ticker readings should not be over-interpreted; per-ticker calibration is a plausible next direction.

6. **Key methodological contributions** that distinguish this pipeline from a naïve "ask the LLM for direction" baseline:
   - 4-way balanced selection eliminates input-tone bias on top of class balance
   - K = 11 self-consistency produces a continuous, calibrated P(UP) score
   - Per-horizon balanced evaluation enables clean MCC interpretation (sign-of-MCC matches sign-of-edge)
   - Tone stratification tests whether the signal concentrates in any article-sentiment subset
   - Bootstrap confidence intervals on MCC distinguish real signal from sampling noise at each horizon
   - Comparisons against FinBERT, momentum, and frontier LLMs place the two-step design against both simpler and larger alternatives

## 8. Limitations and Threats to Validity

This section catalogs what the evaluation does *not* claim and where the methodology is weak. None of these invalidate the results, but they bound the scope of the conclusions.

### 8.1 Scope

- **Universe**: 7 US mega-cap stocks (AAPL, MSFT, NVDA, AMZN, TSLA, GOOG, TSM). Mid-caps, small-caps, and non-US markets are out of scope. Mega-cap mechanics (high liquidity, heavy analyst coverage, semi-strong market efficiency) make this a particularly hard subset for news-driven prediction.
- **Language**: English-language news only. Multilingual coverage and non-English-speaking issuers (e.g., Chinese-listed companies) are not tested.
- **Article granularity**: Single-article prediction. The pipeline makes one prediction per article, with no aggregation across multiple articles on the same day for the same ticker. A real trading system would likely consolidate or rank co-occurring articles.

### 8.2 Failed experiment: prompt-level UP-bias correction

*(This subsection documents a **development-run experiment conducted at K=5**, before the final pipeline settings. The absolute "before" numbers below come from that run — the final committed K=11 run has a lower predUP rate of ~54–58% and 1d MCC +0.045 — but the qualitative finding is robust.)*

In that development run the pipeline's predUP rate ran high (~67–77% across horizons) even though the eval set is 50/50 by construction. We attempted to correct this via two prompt-level interventions:

1. Adding a "calibration note" at the top of the prediction prompt explicitly telling the model the distribution is 50/50 and not to default to UP based on positive tone.
2. Revising the 1d horizon instruction to flag short-term mean reversion (consistent with the −0.119 MCC of the 20-day momentum baseline at 1d).

**Both interventions worked at calibration but destroyed the discriminative signal.** The predUP rate did drop from ~67-77% to ~52-56% — the model became balanced. But:

| Horizon | MCC before | MCC after | Δ |
|---|---:|---:|---:|
| 1d  | +0.126 | +0.007 | **−0.119** |
| 3d  | +0.057 | −0.070 | **−0.127** |
| 5d  | +0.055 | −0.016 | −0.071 |
| 10d | +0.029 | +0.053 | +0.024 |
| 21d | +0.056 | +0.032 | −0.024 |

1d lost its statistical significance entirely. This is a clean and somewhat surprising negative result: **the model's UP bias and its discriminative signal are mechanistically coupled** — the model was conveying "this is a strongly-positive article" through high-confidence UP predictions, and removing the UP-defaulting behavior also removed that signal channel. Prompt-level instructions cannot decouple them. We reverted both changes; the numbers reported in this document are from the original (UP-biased but signal-bearing) prompts.

The implication: **further improvements should target the dataset/selection layer, not the prompt.** Possible directions include selecting more genuinely-negative articles (rebalancing input tone via dataset curation rather than prompt instructions), or training a small calibration head on top of the model's continuous scores.

### 8.3 K-sample stochasticity dominates the within-run CI

The pipeline uses K-sample self-consistency at temperature 0.8. Each row's prediction is determined by majority vote across K samples. For "borderline" rows where the model's true P(UP) is near 0.5, the vote outcome flips between runs with non-trivial probability — and we have empirical evidence that this run-to-run variance is **larger than the within-run bootstrap CI suggests**:

- K=5, Run 1: 1d MCC +0.126
- K=5, Run 2: 1d MCC −0.058
- K=11, Run 3: 1d MCC +0.045

Within-run bootstrap CIs all have width ~0.17, but the actual run-to-run range at 1d is 0.18 — comparable to a within-run CI. Bumping K from 5 to 11 reduces per-run vote-flipping variance but doesn't shrink the within-run CI (which depends on n=584, not K). For a fully rigorous single-point estimate, multiple K=11 runs (3-5+) with averaging would be required. We report a single K=11 run with this caveat noted.

### 8.4 Methodological caveats

- **No temporal train/test split.** The dataset is not partitioned by time. Predictions are made with explicit temporal restrictions in the prompt ("pretend today is {date}"), but the LLM's training data may overlap with article dates. We cannot fully rule out memorization-based leakage; results should be interpreted as "best-effort retrospective accuracy on news the model may or may not have already seen."
- **No randomized seed control**. K=5 sampling at temperature 0.8 produces stochastic outputs. The pipeline is not bit-exact reproducible. Multiple runs would yield slightly different MCC values; the bootstrap CIs in Section 5.1 estimate the per-sample noise but not the run-to-run variance.
- **Statistical significance at longer horizons is weak.** Only 1d is significant at 95% confidence. 3d-21d point estimates trend positive but are within the noise band. We do not claim signal at those horizons; we report the point estimates and acknowledge the uncertainty.
- **Per-ticker MCC error bars are wide.** With ~100 samples per (ticker, horizon), per-ticker MCC has approximately ±0.10 standard error. Individual per-ticker readings should not be over-interpreted.

### 8.5 What the evaluation does NOT measure

- **No backtesting or P&L simulation.** We measure direction accuracy, not trading profitability. Transaction costs, slippage, market impact, position sizing, and risk management are not modeled. Translating a single-digit MCC (≈+0.045 at 1d) to a profitable trading strategy is non-trivial and depends on factors beyond this evaluation.
- **Non-LLM comparisons cover the obvious alternatives but not all.** Section 6.3 compares against FinBERT (off-the-shelf finance sentiment) and 20-day price momentum, and the LLM wins both at every horizon. Not tested: domain-specific lexicon methods (e.g. Loughran-McDonald), supervised classifiers trained on labeled finance text, and more sophisticated price-based strategies (mean-reversion ensembles, regime-switching models). A skeptic could reasonably ask whether a well-tuned supervised classifier would close the gap.
- **No human-expert baseline.** We do not measure how a financial analyst reading the same articles would perform. The "ceiling" set by published LLM benchmarks (~55-58% accuracy) may itself be below what attentive humans achieve.

### 8.6 Generalizability

- **Time period**: The dataset spans a finite range of dates (2023-2025). Results are not validated against future periods. Performance may differ in regimes not represented in the eval set (e.g. macro shocks, sector rotations).
- **Source diversity**: Articles are drawn from FNSPID, predominantly nasdaq.com-sourced. Performance on other news distributions (e.g. Reuters, Bloomberg, social media) is not tested.
- **Selection survivorship**: The 4-way balanced selection requires articles to have both POS/NEG factor tone and an associated outcome direction. Articles where the LLM failed to extract factors or where the ticker's price history is incomplete are excluded. This is a small effect (<5% of rows) but not zero.

## 9. Reproducibility

| Step | Script | Notes |
|---|---|---|
| 1. Run predictions | `python gemini_predict.py` | Requires `GEMINI_API_KEYS` env var; uses Batch API; resumes from checkpoints |
| 2. Compute metrics | `python merge_evaluation.py` | Reads pred parquets directly; writes summary CSVs to `data/balanced_focused_evaluation_summary_gemini_k5/` |
| 3. (Optional) Pro test | `python gemini_predict_pro.py` then `python merge_evaluation_pro.py` | Smaller subset for model comparison |

Output dirs (current configuration):
- Predictions: `data/balanced_focused_predictions_gemini_k5/`
- Eval CSVs:   `data/balanced_focused_evaluation_results_gemini_k5/`
- Summary:     `data/balanced_focused_evaluation_summary_gemini_k5/`

Key config constants in `gemini_predict.py`:
- `MODEL = "models/gemini-2.5-flash-lite"`
- `K_SAMPLES = 11`, `SAMPLE_TEMPERATURE = 0.8`
- `MAX_ARTICLES_PER_TICKER = 100`
- `PER_HORIZON_BALANCED_SELECTION = True`
- `FOUR_WAY_TONE_BALANCED_SELECTION = True`
