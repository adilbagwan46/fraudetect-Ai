# Phase 2A Model Evaluation

## Evaluation discipline

The source is public synthetic PaySim (`SHA-256 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`). Complete-step chronological partitions are training steps 1–323, validation steps 324–377, and held-out test steps 378–743.

Inside training, model fitting uses steps 1–298 (4,010,534 rows; 3,383 fraud) and sigmoid calibration uses later steps 299–323 (453,053 rows; 260 fraud). Preprocessing and class weights are fitted on the model-fit portion only. Validation selects candidate and thresholds. The test file is loaded only after those choices are frozen.

The feature contract is exactly:

```text
transaction_type, amount, origin_balance_before,
hour_of_day, log_amount, amount_to_origin_balance
```

## Validation candidate comparison

All candidates use train-only preprocessing and train-only sigmoid calibration. BALANCED is the maximum-validation-F1 threshold for each candidate.

| Candidate | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN | Review rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic, unweighted | 0.255688 | 0.944547 | 0.713483 | 0.226786 | 0.344173 | 51 | 433 | 0.018870% |
| Logistic, balanced | 0.187044 | 0.984765 | 0.464684 | 0.223214 | 0.301568 | 144 | 435 | 0.028517% |
| HistGradientBoosting, unweighted | **0.896236** | 0.998118 | 0.828729 | 0.803571 | 0.815956 | 93 | 110 | 0.057565% |
| HistGradientBoosting, balanced | 0.891175 | **0.999900** | **0.834879** | **0.803571** | **0.818926** | **89** | **110** | **0.057140%** |

The balanced HistGradientBoosting candidate wins because selection was predefined as highest validation BALANCED-policy F1, with PR-AUC and ROC-AUC only as tie-breakers. Its F1 advantage over the unweighted candidate is small, and the unweighted candidate has slightly higher PR-AUC; both facts are retained rather than hidden.

## Validation review-capacity metrics

Values rank validation transactions by calibrated risk and evaluate exactly the top `floor(N × capacity)` rows.

| Candidate | P@0.1% | R@0.1% | P@0.5% | R@0.5% | P@1.0% | R@1.0% |
|---|---:|---:|---:|---:|---:|---:|
| Logistic, unweighted | 0.191941 | 0.323214 | 0.054707 | 0.460714 | 0.030428 | 0.512500 |
| Logistic, balanced | 0.163309 | 0.275000 | 0.050042 | 0.421429 | 0.034881 | 0.587500 |
| HistGradientBoosting, unweighted | 0.548250 | 0.923214 | 0.118533 | 0.998214 | 0.059266 | 0.998214 |
| HistGradientBoosting, balanced | 0.544008 | 0.916071 | 0.118745 | 1.000000 | 0.059372 | 1.000000 |

## Frozen validation policies

| Mode | Threshold | Precision | Recall | F1 | FP | FN | Review rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIGH_PRECISION | 0.957930 | 1.000000 | 0.362500 | 0.532110 | 0 | 357 | 0.021520% |
| BALANCED (default) | 0.400258 | 0.834879 | 0.803571 | 0.818926 | 89 | 110 | 0.057140% |
| HIGH_RECALL | 0.000750 | 0.299465 | 1.000000 | 0.460905 | 1,310 | 0 | 0.198243% |

HIGH_PRECISION maximizes precision at no more than 0.1% validation review. BALANCED maximizes validation F1. HIGH_RECALL maximizes recall at no more than 1% validation review. BALANCED is the recommended demo default.

## One-shot held-out test result

The frozen BALANCED model and threshold produce:

| Metric | Value |
|---|---:|
| PR-AUC | 0.971447 |
| ROC-AUC | 0.999845 |
| Precision | 0.953802 |
| Recall | 0.828928 |
| F1 | 0.886991 |
| True positives | 3,324 |
| False positives | 161 |
| False negatives | 686 |
| True negatives | 951,573 |
| Review rate | 0.364637% |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[951573, 161],
 [   686, 3324]]
```

Held-out ranking capacity:

| Capacity | Precision | Recall | Reviewed |
|---|---:|---:|---:|
| 0.1% | 1.000000 | 0.238155 | 955 |
| 0.5% | 0.811427 | 0.966833 | 4,778 |
| 1.0% | 0.419483 | 0.999751 | 9,557 |

The illustrative false-positive cost is `161 × ($2.50 review + $5.00 friction) = $1,207.50`. These are demo assumptions, not merchant financial data, and were not used for selection.

## Temporal drift and limitations

Fraud prevalence changes from 0.081616% in training to 0.059367% in validation and 0.419568% in test. The frozen BALANCED review rate consequently rises from 0.057140% on validation to 0.364637% on test. Chronological evaluation intentionally exposes this operational change instead of normalizing it away.

PaySim is synthetic, the six features are deliberately narrow, and probability calibration may not transfer to live merchant traffic. No causal customer-history or genuine device/IP features are used. Results demonstrate a reproducible engineering baseline, not production readiness.
