# Thesis Paper and Code Alignment Notes

Authoritative paper reviewed: `paper/DS10_3CSD - THESIS PAPER.docx`

This file records implementation/documentation alignment issues. It does not modify the original Word document.

## Implemented alignment

| Paper requirement | Active implementation |
|---|---|
| Stratified 5-fold evaluation | `make_5fold_dataset.py` and isolated fold runner |
| Normal vs Subluxation classification | TMD output with Subluxation as positive class |
| Reconstructed connected-attention benchmark | Connected pool3 self-attention projected/fused with final DenseNet features |
| DenseNet201 + CBAM proposed model | Channel and spatial CBAM after final DenseNet feature stage |
| Auxiliary artifact detection | Four classes: none, motion blur, Gaussian noise, metal streak |
| Class-weighted TMD loss | Balanced training sample weights, enabled by default for both models |
| Mean and SD across folds | Per-case summaries and final C1-C4 analyzer |
| Shapiro-Wilk → paired t-test/Wilcoxon | `analyze_chapter4.py`; defaults to the paper's one-tailed `greater` hypothesis and records the alternative |
| Accuracy, precision, recall, specificity, F1 | Computed from held-out fold predictions |
| Inference speed and latency | End-to-end batch pipeline timing after warm-up |
| ECE | Ten-bin ECE stored per fold |
| Grad-CAM | `generate_gradcam.py`, targeting benchmark fused features or proposed post-CBAM refined features |
| Localization Energy and IoU | Available only when an expert ROI CSV is supplied |
| Clean vs artifact-mix cases C1-C4 | Standard result-directory layout and completeness validation |

## Critical paper decisions still required

### 1. Unit of analysis and group leakage

The base study distinguishes **2,135 retained four-frame source radiographs** from **3,425 extracted open-mouth TMJ images**. Its published discussion explicitly states that five-fold cross-validation was image-wise rather than patient-wise and acknowledges possible patient-level leakage. The active thesis code currently receives only the 3,425 extracted images.

For exact methodological replication, the absence of patient grouping must be disclosed because it matches the base study's acknowledged image-wise limitation. For the stronger primary thesis evaluation, recover extraction provenance when available and prepare `source_path,group_id` metadata for `--groups_csv`, where `group_id` identifies the original panorama or patient. Exact-image deduplication is already implemented, but it cannot infer patient identity.

### 2. Conflicting sampling descriptions

The Research Methods section says the complete pool is reorganized into stratified 5-fold cross-validation. The Sampling section also says the exact original 2,191/547/687 train/validation/test split is retained and “applied” to 5-fold CV. These are different designs.

Recommended final wording: combine the source pool, form patient/original-study-grouped outer folds, derive validation only from each outer training partition, and hold each outer test fold untouched. The old fixed split counts may be reported as the source dataset's original organization, not as the final experiment split.

### 3. Duplicate and label audit

The current local working copy contains exact duplicate content, including eight exact-content groups carrying conflicting labels. `data_5_fold/duplicate_audit.csv` identifies them. It has not yet been established whether these issues exist in the complete official Figshare archive or arose in the preparation of the local fixed split. The research team/domain expert must correct those labels or formally approve excluding all ambiguous groups. The decision and resulting final sample count must be reported.

### 4. Acronym consistency

Use `AGML-DenseCBAM` consistently in code and repository documentation because `AGML` is the acronym defined in the thesis and used by the WSL project directory. Update any remaining thesis figures, captions, or filenames that still say `AGMTL`.

### 5. CBAM terminology

CBAM is not only a Spatial Attention Module. It sequentially contains channel attention and spatial attention. Recommended wording: **“CBAM-based attention module comprising channel and spatial attention.”** Use `SAM` only when referring specifically to CBAM's spatial submodule.

### 6. Synthetic versus real clinical artifacts

The implementation creates controlled synthetic corruptions. It does not detect independently annotated real clinical artifacts and does not perform physical metal-artifact reconstruction. Replace unqualified claims such as “clinical artifact detection” with “synthetic artifact-type classification” where appropriate, and avoid claiming demonstrated real-world clinical reliability without external validation.

### 7. Augmentation library mismatch

The Sampling section states that Albumentations is used. The active reproducible implementation uses OpenCV and NumPy for motion blur, Gaussian noise, and simulated bright streaks. Either revise the paper to name OpenCV/NumPy or deliberately replace/test the implementation with a fixed Albumentations version. Do not claim a library that did not generate the final data.

### 8. Statistical direction

The Descriptive/Statistical Method specifies a one-tailed paired t-test, while other sections simply say paired t-test. The hypotheses are directional (“improvement/superior”), so a one-tailed test is defensible only if prespecified before examining final results. `analyze_chapter4.py` defaults to `greater` and records this choice; `--alternative two-sided` is available if the adviser requires a two-sided analysis.

Normality should be evaluated on the **paired fold-wise differences**, not on each model's raw metric distribution separately. With only five folds, Shapiro-Wilk and inferential tests have very low power, and fold estimates are not fully independent. Report fold values, mean ± SD, 95% confidence intervals, effect sizes, and p-values without treating p-values as definitive proof.

### 9. ROI requirement

The paper states that manually annotated condyle/glenoid-fossa bounding boxes are used for Localization Energy and IoU, but no expert ROI file is present. Those outcomes cannot be claimed until ROI annotations, annotator credentials/procedure, and the prespecified Grad-CAM threshold are documented.

### 10. Hardware comparability

The paper lists an RTX 4050 laptop and an Apple M3 device. C1-C4 inference speed, latency, and paired performance comparisons should be generated on the same final software environment and, for efficiency claims, the same hardware. Do not average speed measurements from heterogeneous devices. Record which device produced every final run.

### 11. Benchmark wording

The verified official v1.0 notebook is byte-for-byte identical to `base/Sacncar-Main.ipynb`. It uses one fixed train/validation/test directory layout and contains no five-fold loop. It also computes attention from `pool3_relu` and immediately overwrites that tensor with `conv5_block32_concat`, so the released attention output does not reach the classifier.

The active code therefore reconstructs the connected benchmark architecture from the publication description; it does not use original benchmark weights or recover the unpublished fold assignments. Continue describing it as a **reconstructed benchmark** and disclose architectural or preprocessing deviations from Sancar et al.

## Paper consistency corrections

- Remove the duplicate sentence introducing the Statement of the Problem questions.
- Renumber hypotheses/questions consistently; the extracted document repeats `(1)` and `(2)` labels.
- Use future tense for planned analysis and past tense only after final runs are completed.
- Reconcile references to “2,135 panoramic images” with “3,425 extracted TMJ images” by naming the sample unit each time.
- Correct spelling/grammar in captions and methods (for example, “spatiol,” “accuried,” and hardware-description errors).
- Verify all 2025/2026 references, DOI metadata, publication year, volume, and article number against publisher pages before submission.
- Ensure equations 3.1–3.12 render visibly in the final Word/PDF export; plain-text extraction currently exposes equation labels without their mathematical content.

## Result status

No existing repository metric is approved as a final Chapter IV result. The old single-fold output predates leakage protection and has a test count inconsistent with the current fold. Final tables must be generated only after:

1. duplicate-label adjudication;
2. patient/original-study grouping where metadata is available;
3. fold regeneration;
4. all C1-C4 runs on identical folds/settings/hardware; and
5. final aggregate and explainability analysis.
