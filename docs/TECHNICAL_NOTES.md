# Technical Notes

## 1. Scope and status

These notes describe the completed Final V2 experiment and its relationship to the thesis method, published base study, released base notebook, and preserved V1 pilot. The thesis document `paper/DS10_3CSD - THESIS PAPER.docx` remains the methodological source of truth and is not modified by this repository.

Final V2 outputs are stored locally under `results/final_v2/`. V1 remains under `results/final_3002_seed42/` and Git tag `pilot-v1`. Generated results and source images are excluded from version control.

## 2. Study and benchmark provenance

The verified base study reports:

- 4,270 initial radiographs;
- 2,135 retained four-panel radiographs;
- 3,425 extracted TMJ images;
- 1,478 Normal and 1,947 Subluxation images after excluding 845 uncertain crops; and
- image-wise five-fold cross-validation, with possible patient-level leakage acknowledged as a limitation.

The file `base/Sacncar-Main.ipynb` is byte-for-byte identical to the official version 1.0 notebook. The released notebook uses a fixed train/validation/test directory arrangement and does not implement the reported five-fold loop. It calculates attention from `pool3_relu` but then overwrites that tensor with `conv5_block32_concat`, so the calculated attention output does not reach the classifier.

The active comparison model is therefore a **reconstructed benchmark**. It connects `pool3_relu` self-attention to the final DenseNet representation before classification. Original fold assignments, patient identifiers, trained weights, extraction provenance, and label-adjudication records were not available; exact replication is not claimed.

## 3. Data audit and fold governance

The local collection contained 3,425 files. SHA-256 exact-content auditing found:

| Audit item | Count |
|---|---:|
| Unique image contents | 3,010 |
| Duplicate-content groups | 361 |
| Additional exact copies | 415 |
| Contradictory-label groups | 8 |
| Files in contradictory-label groups | 17 |
| Final non-conflicting experimental images | 3,002 |

For example, `data/train/normal/0166.jpg` and `data/test/subluxation/1872.jpg` are pixel-identical despite carrying opposite labels.

`scripts/make_5fold_dataset.py` groups exact copies by content hash, excludes repeated copies from the experimental pool, and prevents exact-content groups from crossing splits. Its default conflict policy is `error`. Exclusion of contradictory-label groups requires the explicit `exclude` option and formal documentation or approval before thesis reporting.

Each outer fold contains separate training, validation, and held-out test partitions. Validation is derived only from the outer-training partition. Training data support parameter updates; validation data support learning-rate changes, early stopping, and checkpoint selection; test data enter only final evaluation.

Patient and original-panorama identifiers were unavailable. The design is therefore image-wise and cannot guarantee patient-level isolation.

## 4. Experimental design

| Case | Model | Input condition | Main purpose |
|---|---|---|---|
| C1 | Reconstructed benchmark | Clean | Benchmark clean performance |
| C2 | Reconstructed benchmark | Artifact mix | Benchmark stress-test performance |
| C3 | Proposed AGML-DenseCBAM | Clean | Proposed clean performance |
| C4 | Proposed AGML-DenseCBAM | Artifact mix | Proposed stress-test and auxiliary performance |

All cases use identical fold manifests, hardware, primary training settings, and checkpoint criterion. Corresponding clean and artifact-mix cases differ only in the controlled input condition.

## 5. Model definitions

### 5.1 Reconstructed benchmark

1. ImageNet-pretrained DenseNet201 extracts image features.
2. `pool3_relu` features enter a connected self-attention path.
3. Attention features are downsampled and projected.
4. They are concatenated with `conv5_block32_concat` features.
5. A 1×1 fusion convolution and classification head produce Normal/Subluxation probabilities.

### 5.2 Proposed AGML-DenseCBAM

The proposed model uses a shared ImageNet-pretrained DenseNet201 encoder and two branches:

| Branch | Feature source | Head | Output |
|---|---|---|---|
| Primary TMD | Post-CBAM features | Global average pooling → Dense 1,024 → Dropout 0.5 → Batch normalization → Dense 128 | Normal/Subluxation softmax |
| Auxiliary artifact | Pre-CBAM features | Global average + global maximum pooling → Dense 256 → Dropout 0.3 | Four-class artifact softmax |

CBAM sequentially applies channel and spatial attention. The primary branch remains post-CBAM. The auxiliary branch uses pre-CBAM average and maximum pooling because development-only tests showed that post-CBAM global-average features could suppress sparse metal-streak evidence.

The joint objective uses TMD and artifact loss weights of 1.0 and 0.3, respectively.

## 6. Controlled synthetic artifacts

V1 used a maximum-overlay metal transform that produced no pixel change in 50 of 200 training-only calibration images. V2 replaced that transform before final evaluation and applied the same corrected distribution to both models.

| Category | Final V2 definition |
|---|---|
| None | Unmodified image |
| Motion blur | Horizontal kernel sampled from 5, 7, or 9 |
| Gaussian noise | Sigma sampled uniformly from 8 to 12 |
| Metal streak | 1–2 additive full-width localized blurred streaks; thickness 2–4 px; intensity increment 80–110; blur sigma 1.5–2.5 |

The four categories are sampled uniformly. Training corruption is deterministic from the seed, sample, and epoch. Validation and test corruption is deterministic from the seed and sample.

These transformations are controlled computational stress tests. They are not physically validated reconstructions or independently annotated clinical artifacts.

## 7. Locked V2 training settings

| Parameter | Value |
|---|---:|
| Experimental pool | 3,002 unique, non-conflicting images |
| Cross-validation | Five image-wise outer folds |
| Validation fraction | 15% of each outer-training partition |
| Input | 224 × 224 RGB |
| Preprocessing | DenseNet ImageNet preprocessing to `[-1, 1]` |
| Training augmentation | Seeded horizontal flip, probability 0.5 |
| Batch size | 8 |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Maximum epochs | 50 |
| Early-stopping patience | 5 |
| Learning-rate reduction | Factor 0.1; patience 3; minimum `1e-6` |
| Dense-layer L2 | `1e-2` |
| TMD class weighting | Balanced training weights |
| TMD loss weight | 1.0 |
| Artifact loss weight | 0.3 |
| Seed | 42 |
| Precision | Mixed precision |
| Checkpoint monitor | Primary validation TMD loss |

Training TMD loss is class-weighted, while validation TMD loss is unweighted. Their numerical gap must not be interpreted solely as evidence of overfitting.

## 8. V2 development decisions

Development-only Fold 1 smoke tests were used before the final five-fold run:

| Auxiliary configuration | Epochs | TMD accuracy | Artifact accuracy | Macro-F1 | Decision |
|---|---:|---:|---:|---:|---|
| Post-CBAM, loss 0.1 | 5 | 88.02% | 24.96% | 9.99% | Rejected; one-class collapse |
| Post-CBAM, loss 0.3 | 8 | 89.02% | 73.04% | 65.75% | Rejected; clean/metal collapse |
| Pre-CBAM average+maximum pooling, loss 0.3 | 8 | 87.35% | 96.67% | 96.64% | Accepted before final evaluation |

The final setting was selected from development behavior, not by repeatedly inspecting the final five-fold test results.

## 9. Reproducibility controls

The implementation records or validates:

- source-image and fold-manifest hashes;
- exact-content overlap across train, validation, and test partitions;
- seed and deterministic corruption rules;
- model, scenario, and training configuration;
- runner and bundled training-source SHA-256 fingerprints;
- checkpoint and Grad-CAM input fingerprints;
- held-out prediction paths and split identity;
- fold-level histories, predictions, metrics, and confusion matrices; and
- matching fold/configuration requirements before aggregate analysis.

The isolated runner starts each fold in a fresh process. Existing fold outputs are skipped only when their recorded fingerprints match the active configuration. New runs fingerprint the training entry point together with every module under `scripts/pipeline/`, so a change in any training component changes the recorded source fingerprint.

The preserved Final V2 outputs were generated before the readability refactor and retain the source fingerprints recorded at that time. Those outputs are not rewritten. Commit `9bced5a` preserves the exact pre-refactor source corresponding to the finalized repository state, while the modular implementation preserves the same deterministic artifact behavior, model structures, layer names, command-line interface, and output contracts for new reproductions.

## 10. Final V2 results

Values are five-fold mean ± sample standard deviation. ECE is shown as a percentage and lower is better.

| Case | Accuracy | Precision | Recall | Specificity | F1-score | ECE |
|---|---:|---:|---:|---:|---:|---:|
| C1 Benchmark Clean | 89.21 ± 2.20 | 90.93 ± 3.42 | 89.29 ± 1.84 | 89.11 ± 4.35 | 90.07 ± 1.96 | 6.61 ± 2.18 |
| C2 Benchmark Artifact Mix | 88.94 ± 1.00 | 89.95 ± 1.88 | 89.90 ± 2.59 | 87.79 ± 2.66 | 89.89 ± 1.00 | 5.54 ± 1.57 |
| C3 Proposed Clean | 89.44 ± 1.57 | 90.88 ± 1.67 | 89.71 ± 1.44 | 89.11 ± 2.03 | 90.29 ± 1.44 | **4.24 ± 2.02** |
| C4 Proposed Artifact Mix | **90.27 ± 1.44** | **90.85 ± 2.15** | **91.48 ± 1.27** | **88.82 ± 2.83** | **91.15 ± 1.26** | 4.68 ± 1.20 |

### 10.1 Fold-level accuracy and F1-score

| Fold | C1 Acc. | C1 F1 | C2 Acc. | C2 F1 | C3 Acc. | C3 F1 | C4 Acc. | C4 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 88.02 | 89.25 | 88.19 | 89.36 | 88.02 | 89.02 | 88.52 | 89.66 |
| 2 | 88.33 | 89.20 | 88.83 | 89.89 | 90.50 | 91.32 | 91.17 | 92.05 |
| 3 | 89.17 | 90.14 | 88.67 | 89.73 | 87.50 | 88.48 | 89.33 | 90.33 |
| 4 | 87.52 | 88.37 | 88.35 | 88.92 | 90.18 | 90.96 | 90.18 | 90.96 |
| 5 | 93.00 | 93.40 | 90.67 | 91.54 | 91.00 | 91.67 | 92.17 | 92.76 |

### 10.2 Confidence intervals

The intervals use the t-distribution across five folds.

| Case | Accuracy 95% CI | F1-score 95% CI |
|---|---:|---:|
| C1 | 86.47–91.94 | 87.64–92.51 |
| C2 | 87.70–90.18 | 88.65–91.13 |
| C3 | 87.49–91.39 | 88.50–92.08 |
| C4 | 88.48–92.07 | 89.59–92.72 |

### 10.3 Paired model comparisons

The comparison unit is the matched fold (`n = 5`). Differences are proposed minus benchmark. The prespecified alternative is one-tailed (`greater`) at α = 0.05. Shapiro–Wilk is applied to paired differences; all reported p-values exceeded 0.05, so paired t-tests were selected under the prespecified rule. With five pairs, this check has low power and does not prove normality.

| Condition and metric | Mean difference | Shapiro W | Shapiro p | t | One-tailed p | Cohen's `dz` | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Clean accuracy | +0.23 points | 0.8853 | 0.3341 | 0.243 | 0.4100 | 0.109 | Not significant |
| Clean F1 | +0.22 points | 0.8536 | 0.2062 | 0.241 | 0.4108 | 0.108 | Not significant |
| Artifact accuracy | **+1.33 points** | 0.9535 | 0.7619 | 3.613 | **0.0112** | **1.616** | Significant, unadjusted |
| Artifact precision | +0.90 points | 0.9861 | 0.9645 | 1.301 | 0.1315 | 0.582 | Not significant |
| Artifact recall | +1.58 points | 0.9160 | 0.5042 | 1.603 | 0.0921 | 0.717 | Not significant |
| Artifact specificity | +1.03 points | 0.9834 | 0.9517 | 1.071 | 0.1722 | 0.479 | Not significant |
| Artifact F1 | **+1.26 points** | 0.9036 | 0.4300 | 3.396 | **0.0137** | **1.519** | Significant, unadjusted |

The artifact-condition accuracy and F1 results are significant only under the prespecified unadjusted analysis. Neither crosses a conservative Bonferroni threshold of 0.01 if all five metrics form one family. Clean precision, recall, and specificity were also not significant.

Separate robustness-gain comparisons did not reach `p < 0.05`: accuracy `p = 0.1180` and F1 `p = 0.0967`. Robustness superiority should therefore be described as a favorable descriptive trend, not a statistically significant result.

### 10.4 Performance by artifact category

| Category | C2 benchmark accuracy | C4 proposed accuracy | Difference |
|---|---:|---:|---:|
| None | 89.87% | **92.31%** | +2.44 points |
| Motion blur | **89.80%** | 88.23% | −1.57 points |
| Gaussian noise | 87.78% | **89.50%** | +1.72 points |
| Metal streak | 88.30% | **91.17%** | +2.87 points |

These comparisons are descriptive; separate inferential tests were not prespecified.

### 10.5 Auxiliary and efficiency results

C4 artifact accuracy was **80.67% ± 9.72%**, and artifact macro-F1 was **77.57% ± 12.11%**. Pooled recalls were 70.12% for none, 99.87% for motion blur, 100% for Gaussian noise, and 50.63% for metal streak. V2 removed the global one-class collapse, but clean/metal confusion remained fold-dependent.

| Case | Throughput (images/s) | Latency (ms/image) |
|---|---:|---:|
| C1 | 10.85 ± 1.24 | 93.14 ± 10.58 |
| C2 | **12.55 ± 0.28** | **79.68 ± 1.78** |
| C3 | 11.11 ± 2.11 | 93.27 ± 21.70 |
| C4 | 8.01 ± 0.43 | 125.09 ± 6.67 |

Under artifact mix, the proposed model had approximately 36.2% lower throughput and 57.0% higher latency. No speed advantage is claimed.

## 11. Explainability and appendix outputs

Final V2 checkpoints support Grad-CAM at:

| Model | Target layer |
|---|---|
| Reconstructed benchmark | `benchmark_fusion_conv` |
| Proposed model | `cbam_attention` |

The appendix generator selects balanced held-out examples using a stable SHA-256 key independent of prediction correctness, confidence, and heatmap appearance. It produced 40 individual panels and 20 paired benchmark/proposed panels. Metadata include class labels, probabilities, artifact condition, paths, selection key, and hashes for the source image, model input, checkpoint, prediction table, and generation script.

Grad-CAM is qualitative. Localization Energy and intersection-over-union must not be reported without independently prepared expert ROI annotations, documented annotator credentials and procedure, and a prespecified threshold.

## 12. Reporting language and limitations

Use the following descriptions consistently:

- **Study type:** quantitative, retrospective, non-interventional, secondary-data computational model-development and comparative evaluation study.
- **Benchmark:** reconstructed DenseNet201 attention benchmark.
- **Attention module:** CBAM-based attention comprising channel and spatial attention.
- **Artifacts:** controlled synthetic artifacts or synthetic artifact types, not clinical artifacts.
- **System status:** research prototype, not a clinically validated diagnostic system.

The principal limitations are:

1. contradictory-label exclusion still requires formal governance documentation;
2. patient/original-panorama grouping is impossible without provenance metadata;
3. synthetic artifacts do not establish clinical or acquisition-physics realism;
4. five folds provide low-powered inferential and normality tests;
5. one-tailed tests and multiplicity must be disclosed;
6. Grad-CAM does not establish causal or anatomically valid reasoning; and
7. the released base implementation does not permit exact reproduction of the published experiment.

The relevant institutional review body must determine whether secondary use of the existing de-identified human radiographs is exempt or otherwise reviewable. Approval must not be implied retroactively.

## 13. Thesis consistency items

Before submission:

- distinguish 2,135 retained source radiographs from 3,425 extracted TMJ images;
- describe the fixed source split only as historical organization, not as the final five-fold design;
- name OpenCV and NumPy as the implemented corruption tools rather than Albumentations;
- state that normality was assessed on paired fold-wise differences;
- report fold values, mean ± SD, confidence intervals, p-values, and effect sizes together;
- use the same hardware for comparative efficiency claims;
- include ROI metrics only if independent expert annotations are obtained; and
- retain the qualified result: artifact-condition accuracy and F1 improved under the prespecified unadjusted test, while other superiority claims were unsupported.

## 14. Output inventory

Each trained fold produces:

```text
fold_N_history.csv
fold_N_learning_curves.png
fold_N_confusion_matrix.csv
fold_N_artifact_confusion_matrix.csv   # proposed model
fold_N_predictions.csv
fold_N_results.csv
fold_N_<model>_<scenario>_best.keras
run_config.json
```

Aggregate analysis is written to `results/final_v2/analysis/`. Learning curves and Grad-CAM panels can be regenerated from preserved histories and checkpoints without retraining or changing held-out predictions.
