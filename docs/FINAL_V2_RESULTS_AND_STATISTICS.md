# Final V2 Results, Changes, and Statistical Analysis

## 1. Scope and reporting status

This document summarizes the completed V2 five-fold experiment stored under:

```text
chapter4_results/final_v2/
```

V2 used 3,002 unique, non-conflicting images after exact-content auditing and documented exclusion of unresolved conflicting-label groups. Every case used the same five image-wise folds, seed, preprocessing, batch size, maximum epochs, optimizer settings, hardware, and integrity checks.

These results are separate from the preserved V1 pilot in `chapter4_results/final_3002_seed42/`. Patient identifiers and original-panorama provenance were unavailable; therefore, exact-image leakage was prevented, but patient-level independence cannot be guaranteed. Synthetic artifacts are controlled stress tests rather than clinically validated acquisition-physics simulations.

## 2. Experimental cases

| Case | Model | Input condition | Primary purpose |
|---|---|---|---|
| C1 | Reconstructed DenseNet201 attention benchmark | Clean | Benchmark clean performance |
| C2 | Reconstructed DenseNet201 attention benchmark | V2 artifact mix | Benchmark robustness |
| C3 | Proposed AGML-DenseCBAM V2 | Clean | Proposed clean performance |
| C4 | Proposed AGML-DenseCBAM V2 | V2 artifact mix | Proposed robustness and auxiliary artifact classification |

## 3. What changed in V2

### 3.1 Dataset governance and fold safety

| Earlier limitation | V2 implementation |
|---|---|
| Duplicate images could occur across source folders | SHA-256 exact-content auditing and deduplication |
| Contradictory labels existed for pixel-identical images | Strict conflict detection; 8 conflicting groups involving 17 files excluded under the documented conflict policy |
| Released base implementation did not provide the reported five-fold loop | Reproducible stratified five-fold generation with fold manifests |
| Original patient/study identifiers were unavailable | Content-hash grouping prevents exact-image leakage; patient-level limitation remains disclosed |
| Existing results could be reused after code or fold changes | Fold-manifest, runner-script, and training-script fingerprints prevent stale-result reuse |

### 3.2 Benchmark reconstruction

The published/released base notebook calculated an intermediate attention tensor but then overwrote it with the final DenseNet feature tensor. The active benchmark was therefore reconstructed so that its `pool3_relu` self-attention output is connected to the classifier:

1. DenseNet201 `pool3_relu` features enter self-attention.
2. The attention features are downsampled and projected.
3. They are concatenated with `conv5_block32_concat` features.
4. A 1×1 fusion convolution and classification head produce the TMD prediction.

Accordingly, the benchmark is described as a **reconstruction from the publication**, not a direct reproduction using the authors' original folds or trained weights.

### 3.3 Corrected V2 synthetic-artifact protocol

The V1 metal-streak maximum-overlay transform produced no pixel change in 50 of 200 training-only calibration samples. V2 replaced it with a guaranteed additive localized transform and moderated the other corruption ranges.

| Artifact category | Final V2 parameters |
|---|---|
| None | Unmodified image |
| Motion blur | Horizontal kernel sampled from 5, 7, or 9 |
| Gaussian noise | Sigma sampled uniformly from 8 to 12 |
| Metal streak | 1–2 additive full-width localized blurred streaks; thickness 2–4 px; intensity increment 80–110; blur sigma 1.5–2.5 |

The four categories remained uniformly sampled. Training artifacts were deterministic by seed, sample, and epoch; validation/test artifacts were deterministic by seed and sample. Benchmark and proposed models received identical corruption rules.

### 3.4 Corrected proposed multi-task architecture

| Component | Final V2 design |
|---|---|
| Shared encoder | ImageNet-pretrained DenseNet201, fully fine-tuned |
| Primary TMD branch | Post-CBAM features → global average pooling → Dense 1,024 → Dropout 0.5 → BatchNorm → Dense 128 → two-class softmax |
| Auxiliary artifact branch | **Pre-CBAM** features → global average pooling + global max pooling → concatenation → Dense 256 → Dropout 0.3 → four-class softmax |
| TMD loss weight | 1.0 |
| Artifact loss weight | 0.3 |

The auxiliary branch was moved before CBAM because development-only smoke testing showed that post-CBAM global-average features could suppress sparse metal-streak evidence. Global max pooling was added to preserve localized high-intensity evidence, while TMD classification continued to use the CBAM-refined representation.

### 3.5 Training and evaluation safeguards

- DenseNet ImageNet preprocessing: `[0, 255] → [-1, 1]`
- Input size: 224 × 224 RGB
- Batch size: 8
- Adam learning rate: `1e-4`
- L2 strength: `1e-2` on the configured dense layers
- Maximum epochs: 50
- Early stopping patience: 5, monitoring primary validation TMD loss
- ReduceLROnPlateau: factor 0.1, patience 3, minimum learning rate `1e-6`
- Balanced TMD class weighting
- Mixed-precision GPU training
- Seed: 42
- Added ECE, artifact macro-F1, artifact per-class recall, per-artifact TMD accuracy, fold confusion matrices, throughput, and latency

## 4. Final V2 five-fold scores

Values are five-fold mean ± sample standard deviation. Accuracy-related values are percentages.

| Case | Accuracy | Precision | Recall | Specificity | F1-score | ECE |
|---|---:|---:|---:|---:|---:|---:|
| C1 Benchmark Clean | 89.21 ± 2.20 | 90.93 ± 3.42 | 89.29 ± 1.84 | 89.11 ± 4.35 | 90.07 ± 1.96 | 6.61 ± 2.18 |
| C2 Benchmark Artifact Mix | 88.94 ± 1.00 | 89.95 ± 1.88 | 89.90 ± 2.59 | 87.79 ± 2.66 | 89.89 ± 1.00 | 5.54 ± 1.57 |
| C3 Proposed Clean | 89.44 ± 1.57 | 90.88 ± 1.67 | 89.71 ± 1.44 | 89.11 ± 2.03 | 90.29 ± 1.44 | **4.24 ± 2.02** |
| C4 Proposed Artifact Mix | **90.27 ± 1.44** | **90.85 ± 2.15** | **91.48 ± 1.27** | **88.82 ± 2.83** | **91.15 ± 1.26** | 4.68 ± 1.20 |

ECE is displayed as a percentage for readability; lower is better.

### 4.1 Fold-level primary scores

| Fold | C1 Accuracy | C1 F1 | C2 Accuracy | C2 F1 | C3 Accuracy | C3 F1 | C4 Accuracy | C4 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 88.02 | 89.25 | 88.19 | 89.36 | 88.02 | 89.02 | 88.52 | 89.66 |
| Fold 2 | 88.33 | 89.20 | 88.83 | 89.89 | 90.50 | 91.32 | 91.17 | 92.05 |
| Fold 3 | 89.17 | 90.14 | 88.67 | 89.73 | 87.50 | 88.48 | 89.33 | 90.33 |
| Fold 4 | 87.52 | 88.37 | 88.35 | 88.92 | 90.18 | 90.96 | 90.18 | 90.96 |
| Fold 5 | 93.00 | 93.40 | 90.67 | 91.54 | 91.00 | 91.67 | 92.17 | 92.76 |

### 4.2 Accuracy and F1 95% confidence intervals

The intervals use the t-distribution across five folds.

| Case | Accuracy mean | Accuracy 95% CI | F1 mean | F1 95% CI |
|---|---:|---:|---:|---:|
| C1 Benchmark Clean | 89.21 | 86.47–91.94 | 90.07 | 87.64–92.51 |
| C2 Benchmark Artifact Mix | 88.94 | 87.70–90.18 | 89.89 | 88.65–91.13 |
| C3 Proposed Clean | 89.44 | 87.49–91.39 | 90.29 | 88.50–92.08 |
| C4 Proposed Artifact Mix | 90.27 | 88.48–92.07 | 91.15 | 89.59–92.72 |

## 5. Performance by artifact category

These values are mean TMD accuracies across folds; they measure disease classification under each deterministic synthetic-artifact category.

| Artifact category | C2 Benchmark TMD accuracy | C4 Proposed TMD accuracy | Proposed − Benchmark |
|---|---:|---:|---:|
| None | 89.87% | **92.31%** | +2.44 points |
| Motion blur | **89.80%** | 88.23% | −1.57 points |
| Gaussian noise | 87.78% | **89.50%** | +1.72 points |
| Metal streak | 88.30% | **91.17%** | +2.87 points |

These per-artifact differences are descriptive; no separate inferential tests were prespecified for them.

## 6. Auxiliary artifact-classification results

The C4 auxiliary head produced:

- Artifact accuracy: **80.67% ± 9.72%**
- Artifact macro-F1: **77.57% ± 12.11%**

| Artifact class | Fold-mean recall ± SD | Pooled correct / support | Pooled recall |
|---|---:|---:|---:|
| None | 70.42% ± 36.19% | 535 / 763 | 70.12% |
| Motion blur | 99.86% ± 0.31% | 757 / 758 | 99.87% |
| Gaussian noise | 100.00% ± 0.00% | 770 / 770 | 100.00% |
| Metal streak | 52.61% ± 40.32% | 360 / 711 | 50.63% |

The V1 global collapse was corrected: the final V2 auxiliary head did not classify every sample as a single category. However, fold-dependent confusion between the clean and metal-streak categories remained substantial. This must be reported as a limitation. The C3 clean-only artifact macro-F1 of 25% is not substantively meaningful because only the `none` class is present in that condition.

## 7. Efficiency results

| Case | Throughput (images/s) | Latency (ms/image) |
|---|---:|---:|
| C1 Benchmark Clean | 10.85 ± 1.24 | 93.14 ± 10.58 |
| C2 Benchmark Artifact Mix | **12.55 ± 0.28** | **79.68 ± 1.78** |
| C3 Proposed Clean | 11.11 ± 2.11 | 93.27 ± 21.70 |
| C4 Proposed Artifact Mix | 8.01 ± 0.43 | 125.09 ± 6.67 |

Under the artifact-mix condition, the proposed model had approximately 36.2% lower throughput and 57.0% higher latency than the benchmark. V2 therefore supports an artifact-condition accuracy/F1 advantage but not an overall speed advantage.

## 8. Paired model-comparison statistics

### 8.1 Statistical procedure

- Unit of pairing: matching fold
- Number of pairs: 5
- Difference direction: proposed minus benchmark
- Alternative hypothesis: proposed model is greater
- Alpha: 0.05
- Normality check: Shapiro–Wilk test on paired differences
- Selected test: paired t-test for all reported comparisons
- Effect size: Cohen's `dz`

Every Shapiro–Wilk p-value was greater than 0.05, so the analysis selected the paired t-test under its prespecified decision rule. This does not prove normality: because there are only five paired folds, Shapiro–Wilk tests and inferential p-values have low power. The reported tests are **one-tailed and unadjusted**. If all five metrics within a condition are treated as one multiple-testing family, a conservative Bonferroni threshold would be 0.01; neither artifact-condition p-value would cross that stricter threshold. The significant findings below therefore apply to the prespecified unadjusted analysis and should not be generalized to every metric.

### 8.2 Clean condition: C3 proposed versus C1 benchmark

| Metric | Shapiro W | Shapiro p | Mean difference | t statistic | One-tailed p | Cohen's dz | Significant at 0.05? |
|---|---:|---:|---:|---:|---:|---:|---|
| Accuracy | 0.8853 | 0.3341 | +0.23 points | 0.243 | 0.4100 | 0.109 | No |
| Precision | 0.8310 | 0.1416 | −0.05 points | −0.049 | 0.5184 | −0.022 | No |
| Recall | 0.8693 | 0.2637 | +0.43 points | 0.319 | 0.3829 | 0.143 | No |
| Specificity | 0.8938 | 0.3765 | −0.003 points | −0.003 | 0.5010 | −0.001 | No |
| F1-score | 0.8536 | 0.2062 | +0.22 points | 0.241 | 0.4108 | 0.108 | No |

**Interpretation:** V2 provides no statistically significant evidence that the proposed model improves clean-condition performance over the reconstructed benchmark.

### 8.3 Artifact-mix condition: C4 proposed versus C2 benchmark

| Metric | Shapiro W | Shapiro p | Mean difference | t statistic | One-tailed p | Cohen's dz | Significant at 0.05? |
|---|---:|---:|---:|---:|---:|---:|---|
| Accuracy | 0.9535 | 0.7619 | **+1.33 points** | 3.613 | **0.0112** | **1.616** | **Yes** |
| Precision | 0.9861 | 0.9645 | +0.90 points | 1.301 | 0.1315 | 0.582 | No |
| Recall | 0.9160 | 0.5042 | +1.58 points | 1.603 | 0.0921 | 0.717 | No |
| Specificity | 0.9834 | 0.9517 | +1.03 points | 1.071 | 0.1722 | 0.479 | No |
| F1-score | 0.9036 | 0.4300 | **+1.26 points** | 3.396 | **0.0137** | **1.519** | **Yes** |

**Interpretation:** Under the prespecified unadjusted one-tailed analysis, the proposed model significantly outperformed the benchmark in artifact-mix **accuracy and F1-score**, with large paired standardized effects. Precision, recall, and specificity were not individually significant.

## 9. Paired robustness-change statistics

A degradation value is clean performance minus artifact-mix performance. A negative degradation means the artifact-mix score was higher than the clean score. The robustness-gain difference is benchmark degradation minus proposed degradation; positive values favor the proposed model.

| Metric | Shapiro W | Shapiro p | Benchmark mean degradation | Proposed mean degradation | Robustness-gain difference | One-tailed p | Cohen's dz | Significant? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Accuracy | 0.9279 | 0.5824 | +0.27 points | −0.83 points | +1.10 points | 0.1180 | 0.623 | No |
| Precision | 0.9001 | 0.4107 | +0.98 points | +0.03 points | +0.95 points | 0.2918 | 0.266 | No |
| Recall | 0.8938 | 0.3767 | −0.61 points | −1.77 points | +1.16 points | 0.1101 | 0.649 | No |
| Specificity | 0.8788 | 0.3041 | +1.33 points | +0.29 points | +1.04 points | 0.3208 | 0.225 | No |
| F1-score | 0.9173 | 0.5128 | +0.18 points | −0.86 points | +1.04 points | 0.0967 | 0.698 | No |

**Interpretation:** The proposed model did not show average accuracy or F1 degradation under artifact mix, whereas the benchmark showed small decreases. However, the separate paired robustness-gain tests did not reach `p < 0.05`; robustness superiority should therefore be described as a favorable descriptive trend rather than a statistically significant finding.

## 10. V1 pilot compared with final V2

This comparison is descriptive because V1 and V2 used different artifact transformations and proposed-model feature routing. It should not be presented as a controlled significance test.

| Case | V1 accuracy | V2 accuracy | Change | V1 F1 | V2 F1 | Change |
|---|---:|---:|---:|---:|---:|---:|
| C1 Benchmark Clean | 89.21% | 89.21% | 0.00 | 90.07% | 90.07% | 0.00 |
| C2 Benchmark Artifact Mix | 89.24% | 88.94% | −0.30 | 90.08% | 89.89% | −0.19 |
| C3 Proposed Clean | 89.54% | 89.44% | −0.10 | 90.24% | 90.29% | +0.05 |
| C4 Proposed Artifact Mix | 87.74% | 90.27% | **+2.53** | 88.45% | 91.15% | **+2.70** |

The principal corrective outcome was recovery of C4 TMD performance and removal of the global auxiliary collapse. This does not prove that the synthetic artifacts are clinically realistic.

## 11. Thesis-ready conclusion

> Across five matched image-wise folds, the proposed AGML-DenseCBAM V2 model achieved 90.27% ± 1.44% accuracy and 91.15% ± 1.26% F1-score under the controlled artifact-mix condition, compared with 88.94% ± 1.00% accuracy and 89.89% ± 1.00% F1-score for the reconstructed DenseNet201 attention benchmark. Prespecified unadjusted one-tailed paired t-tests indicated significant improvements in artifact-condition accuracy (`p = 0.0112`, Cohen's `dz = 1.616`) and F1-score (`p = 0.0137`, Cohen's `dz = 1.519`). Clean-condition differences and the separate robustness-gain comparisons were not statistically significant. Consequently, the findings support a qualified artifact-condition advantage in accuracy and F1-score, but not superiority across all metrics. Interpretation remains limited by the five-fold sample size, image-wise rather than patient-wise splitting, absence of original provenance identifiers, and use of controlled synthetic rather than clinically validated artifacts.

## 12. Source files

The values in this document were taken from:

```text
chapter4_results/final_v2/*/all_fold_results.csv
chapter4_results/final_v2/*/summary_mean_std.csv
chapter4_results/final_v2/analysis/chapter4_summary_mean_sd_ci95.csv
chapter4_results/final_v2/analysis/paired_model_comparisons.csv
chapter4_results/final_v2/analysis/paired_robustness_comparisons.csv
chapter4_results/final_3002_seed42/*/all_fold_results.csv
```
