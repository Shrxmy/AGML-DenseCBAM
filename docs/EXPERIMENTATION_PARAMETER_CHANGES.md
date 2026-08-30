# Experimentation Parameters, Changes, and Outcomes

## 1. Purpose

This document records why parameters or architecture details changed across the preserved V1 pilot, completed V2 experiment, and exploratory V3 training extension. It distinguishes development-only observations from final outer-fold results so that unsuccessful settings are not hidden and final test outcomes are not used as tuning targets.

- V1 pilot: Git tag `pilot-v1`; results in `chapter4_results/final_3002_seed42/`
- Final V2: branch/tag history under `artifact-v2`; results in `chapter4_results/final_v2/`
- Exploratory V3: branch `experimental-v3`; development validation only at present

## 2. Parameters held constant in the V1 and V2 final comparisons

| Parameter | Locked value |
|---|---:|
| Experimental images | 3,002 unique, non-conflicting images |
| Cross-validation | Five image-wise outer folds |
| Validation | Derived only from each outer training partition |
| Input size | 224 × 224 RGB |
| Backbone | ImageNet-pretrained DenseNet201 |
| Batch size | 8 |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Maximum epochs | 50 |
| Early-stopping patience | 5 |
| LR reduction | Factor 0.1; patience 3; minimum `1e-6` |
| Dense-layer L2 strength | `1e-2` |
| TMD loss weight | 1.0 |
| Artifact loss weight | 0.3 |
| TMD class weighting | Balanced training weights |
| Random seed | 42 |
| Precision | Mixed precision on RTX 4050 GPU |
| Primary checkpoint monitor | Validation TMD loss |

## 3. V1-to-V2 corrective changes

| Component | V1 pilot | Final V2 | Reason for change |
|---|---|---|---|
| Motion-blur kernel | 5, 7, 9, or 11 | 5, 7, or 9 | Remove the strongest blur extreme while retaining moderate degradation |
| Gaussian-noise sigma | Uniform 8–18 | Uniform 8–12 | Use a moderate controlled range |
| Metal streak count | 1–3 | 1–2 | Reduce visual severity |
| Metal transform | Maximum-overlay blurred lines | Additive full-width localized blurred streaks | V1 could produce a pixel-identical output |
| Metal thickness | 1–4 px | 2–4 px | Keep localized streaks visible |
| Metal blur sigma | 2.0–5.0 | 1.5–2.5 | Preserve localized evidence |
| Metal intensity | Overlay intensity 150–235 | Additive increment 80–110 | Guarantee change without replacing most anatomy |
| Auxiliary feature source | Shared post-CBAM global-average features | Pre-CBAM final DenseNet features | Post-CBAM features suppressed localized corruption evidence |
| Auxiliary pooling | Global average only | Concatenated global average and global max | Max pooling preserves sparse high-intensity streak evidence |
| Primary TMD feature source | Post-CBAM | Post-CBAM | Preserved because CBAM is the proposed TMD attention mechanism |
| Artifact metrics | Primarily accuracy | Accuracy, macro-F1, per-class recall, and confusion matrix | Detect class collapse hidden by overall accuracy |
| Reproducibility | Seeded runs | Seed plus fold/script/config fingerprints | Prevent stale or incompatible result reuse |

### V1 calibration finding

A training-only audit found that 50 of 200 V1 metal-streak transformations (25%) were pixel-identical to their source image. This made the clean and metal labels internally inconsistent. V1 was preserved as a pilot rather than overwritten.

## 4. Development sequence for the V2 auxiliary branch

These are development/smoke observations, not the final five-fold estimate. The abbreviated runs had different epoch counts and should not be compared as if they were independent final experiments.

| Development configuration | Epochs | TMD accuracy | Artifact accuracy | Artifact macro-F1 | Key per-class finding | Decision |
|---|---:|---:|---:|---:|---|---|
| Post-CBAM auxiliary, loss weight 0.1 | 5 | 88.02% | 24.96% | 9.99% | All samples predicted as Gaussian noise; clean recall 0% | Reject: auxiliary task did not learn |
| Post-CBAM auxiliary, loss weight 0.3 | 8 | 89.02% | 73.04% | 65.75% | Blur/noise/metal recall 100%, but clean recall 0% | Reject branch placement: clean/metal collapse remained |
| **Pre-CBAM average+max auxiliary, weight 0.3** | 8 | 87.35% | **96.67%** | **96.64%** | None 96.30%; blur 100%; noise 100%; metal 89.93% | Accept architecture for locked V2 |

The weight remained 0.3 in final V2. The decisive correction was moving the auxiliary branch before CBAM and combining average/max pooling; the setting was not chosen by observing the final five-fold test outcomes.

## 5. Final V1 and V2 case results

Values are five-fold mean ± sample standard deviation.

| Case | V1 accuracy | V2 accuracy | Accuracy change | V1 F1 | V2 F1 | F1 change |
|---|---:|---:|---:|---:|---:|---:|
| C1 Benchmark Clean | 89.21 ± 2.20 | 89.21 ± 2.20 | 0.00 | 90.07 ± 1.96 | 90.07 ± 1.96 | 0.00 |
| C2 Benchmark Artifact Mix | 89.24 ± 0.83 | 88.94 ± 1.00 | −0.30 | 90.08 ± 0.57 | 89.89 ± 1.00 | −0.19 |
| C3 Proposed Clean | 89.54 ± 1.73 | 89.44 ± 1.57 | −0.10 | 90.24 ± 1.72 | 90.29 ± 1.44 | +0.05 |
| C4 Proposed Artifact Mix | 87.74 ± 2.69 | **90.27 ± 1.44** | **+2.53** | 88.45 ± 2.73 | **91.15 ± 1.26** | **+2.70** |

The V1/V2 comparison is descriptive because the artifact protocol and proposed auxiliary feature routing changed. Formal V2 model-comparison statistics are reported in [`FINAL_V2_RESULTS_AND_STATISTICS.md`](FINAL_V2_RESULTS_AND_STATISTICS.md).

## 6. Final V2 auxiliary results

| Measure | Five-fold result |
|---|---:|
| Artifact accuracy | 80.67% ± 9.72% |
| Artifact macro-F1 | 77.57% ± 12.11% |
| Mean none recall | 70.42% ± 36.19% |
| Mean motion-blur recall | 99.86% ± 0.31% |
| Mean Gaussian-noise recall | 100.00% ± 0.00% |
| Mean metal-streak recall | 52.61% ± 40.32% |

The global one-class collapse was removed, but clean/metal confusion remained fold-dependent and must be reported as a limitation.

## 7. Exploratory V3 parameter extension

V3 was created only after V2 was completed and inspected. It is a separate exploratory experiment and cannot be described as untouched confirmation.

| Parameter | Final V2 | Exploratory V3 |
|---|---|---|
| Backbone training | Full DenseNet201 fine-tuning from epoch 1 | Stage 1 frozen backbone; Stage 2 selective `conv5` fine-tuning |
| Warm-up | None | 5 epochs for the locked full schedule |
| Fine-tune boundary | Entire backbone trainable | From `conv5_block1_0_bn` |
| Fine-tune learning rate | `1e-4` for full model | `1e-5` in Stage 2 |
| Backbone BatchNorm | Trainable with backbone | Frozen during Stage 2 |
| Horizontal flip | Probability 0.5 | Probability 0.5 |
| Rotation | None | Uniform −5° to +5° |
| Translation | None | Uniform −3% to +3% horizontally/vertically |
| Zoom | None | Uniform scale 0.95–1.05 |
| Contrast | None | Uniform factor 0.90–1.10 |
| Border handling | Not applicable | Reflection padding |
| Model heads | Final V2 heads | Unchanged |
| Synthetic-artifact protocol | Final V2 | Unchanged |
| Development evaluation | Final outer folds already complete | Validation only; analyzer rejects these as Chapter IV test results |

### Abbreviated V3 functionality smoke

| Item | Result |
|---|---:|
| Schedule | 3 frozen + 7 selective fine-tuning epochs |
| Evaluation split | Fold 1 validation only |
| TMD accuracy | 69.10% |
| TMD precision | 67.38% |
| TMD recall | 83.96% |
| TMD specificity | 51.28% |
| TMD F1 | 74.76% |
| Artifact accuracy | 90.09% |
| Artifact macro-F1 | 90.03% |
| Best validation TMD loss | 0.61145 |
| Selected stage | Selective fine-tuning |

Training TMD accuracy was 70.36% and validation accuracy was 69.10%, so the abbreviated smoke showed **underfitting rather than overfitting**. Validation TMD loss improved through its final epoch. The locked 5+45 full development schedule must therefore be evaluated before deciding whether V3 is viable; parameters must not be repeatedly changed merely to exceed V2.

## 8. Appendix outputs produced for each new fold

Every newly trained fold now saves:

| Output | Filename pattern |
|---|---|
| Numerical history | `fold_N_history.csv` |
| Training/validation loss figure | `fold_N_learning_curves.png` |
| TMD confusion matrix | `fold_N_confusion_matrix.csv` |
| Artifact confusion matrix, proposed model | `fold_N_artifact_confusion_matrix.csv` |
| Held-out predictions | `fold_N_predictions.csv` |
| Fold metrics | `fold_N_results.csv` |
| Best checkpoint | `fold_N_<model>_<scenario>_best.keras` |

For the proposed model, the learning-curve figure contains total objective loss, primary TMD loss, and auxiliary artifact loss. V3 figures mark the transition from frozen warm-up to selective fine-tuning with a vertical dashed line. Each figure explicitly notes that training TMD loss is class-weighted while validation TMD loss is unweighted; their numerical gap must therefore not be interpreted solely as an overfitting measure.
