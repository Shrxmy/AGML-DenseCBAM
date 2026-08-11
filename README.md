# AGMTL-DenseCBAM

Leakage-resistant experimental pipeline for **Normal vs Subluxation** classification from TMJ panoramic radiograph images. The proposed model combines an ImageNet-pretrained DenseNet201 encoder, CBAM attention, a primary TMD classifier, and an auxiliary synthetic-artifact classifier.

> Research software only. It is not a clinical diagnostic system.

## Important validity status

The original 3,425-file dataset contains exact duplicate images, including duplicates crossing the original train/validation/test folders and a small number carrying conflicting labels. Old folds and the result under `chapter4_results/oldstyle_results/` were produced before leakage protection was added. They are retained only as legacy evidence and **must not be reported as final Chapter IV results**.

The active pipeline now:

- hashes images before splitting;
- deduplicates same-label exact copies;
- rejects conflicting-label copies unless exclusion is explicitly approved;
- keeps exact copies and optional patient/study groups within one split;
- validates train/validation/test integrity before every final run;
- uses deterministic validation/test corruptions;
- applies DenseNet201 ImageNet preprocessing;
- maps multi-task outputs explicitly by name;
- reports the auxiliary artifact accuracy;
- produces paired C1-C4 statistical analysis only when all expected folds exist.

## Active files

| File | Purpose |
|---|---|
| `make_5fold_dataset.py` | Duplicate audit and leakage-resistant fold generation |
| `train_one_case_5fold.py` | Model, training, validation, and per-fold evaluation |
| `run_case_5fold_isolated.py` | Runs one experimental case in a fresh process per fold |
| `analyze_chapter4.py` | C1-C4 summary, plots, paired tests, and robustness comparison |
| `AGMTL-DenseCBAM-Training.ipynb` | Notebook interface to the active scripts |
| `THESIS_CODE_ALIGNMENT.md` | Direct audit of the paper's methodology against the implementation |
| `base/` | Base-study implementation retained for reference only |
| `legacy/` | Prototypes retained for traceability; do not use for final experiments |

## Environment

Recommended: Python 3.10 or 3.11 and an NVIDIA GPU/Google Colab runtime.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_tf_gpu.py
python -m unittest discover -s tests -v
```

TensorFlow GPU support depends on the operating system/CUDA environment. Current native Windows TensorFlow releases may require WSL2 for modern NVIDIA GPU support.

## Expected source data

```text
data/
  train/
    normal/
    subluxation/
  validation/
    normal/
    subluxation/
  test/
    normal/
    subluxation/
```

The original folder assignment is treated only as a source pool. New folds are created from the combined pool.

## 1. Audit and generate safe folds

First run with the default conflict policy:

```bash
python make_5fold_dataset.py \
  --input_root data \
  --output_root data_5_fold \
  --n_splits 5 \
  --val_size 0.15 \
  --seed 42
```

If conflicting labels exist, generation stops after writing:

```text
data_5_fold/duplicate_audit.csv
```

Review those rows with the dataset owner/domain expert. Correcting the source labels is preferred. If the research team formally approves excluding all ambiguous exact-image groups, document that decision and run:

```bash
python make_5fold_dataset.py \
  --input_root data \
  --output_root data_5_fold \
  --n_splits 5 \
  --val_size 0.15 \
  --seed 42 \
  --conflict_policy exclude
```

### Patient/study-level grouping

Exact-image deduplication does not prove patient independence. If patient or original-study IDs are available, create a CSV:

```csv
source_path,group_id
data/train/normal/0001.jpg,patient_001
data/train/subluxation/0002.jpg,patient_002
```

Then add:

```bash
--groups_csv patient_groups.csv
```

All copies of every source image must be mapped. Final medical-imaging claims should clearly state whether patient/study grouping was available.

## 2. Smoke test

Use a separate smoke-test directory so partial results cannot be mixed with final results:

```bash
python run_case_5fold_isolated.py \
  --model_type proposed \
  --scenario artifact_mix \
  --epochs 2 \
  --fold_limit 1 \
  --output_dir chapter4_results/smoke_proposed_artifact_mix
```

Do not use `--skip_integrity_check` for any reported experiment.

## 3. Run the four final cases

Use the same hyperparameters, folds, seed, and hardware for all cases.

```bash
python run_case_5fold_isolated.py --model_type benchmark --scenario clean        --epochs 50 --output_dir chapter4_results/benchmark_clean
python run_case_5fold_isolated.py --model_type benchmark --scenario artifact_mix --epochs 50 --output_dir chapter4_results/benchmark_artifact_mix
python run_case_5fold_isolated.py --model_type proposed  --scenario clean        --epochs 50 --output_dir chapter4_results/proposed_clean
python run_case_5fold_isolated.py --model_type proposed  --scenario artifact_mix --epochs 50 --output_dir chapter4_results/proposed_artifact_mix
```

Useful options:

```text
--batch_size 8
--learning_rate 1e-4
--l2_strength 1e-2
--tmd_loss_weight 1.0
--artifact_loss_weight 0.35
--freeze_backbone
--no-mixed_precision
--no-class_weighting
--skip_existing
```

`--l2_strength` is kernel L2 regularization, not AdamW decoupled weight decay. The old `--weight_decay` name remains only as a compatibility alias.

## 4. Generate Chapter IV analysis

This command deliberately fails if any case does not contain exactly five unique folds:

```bash
python analyze_chapter4.py \
  --results_root chapter4_results \
  --expected_folds 5
```

Outputs are written to `chapter4_results/analysis/`:

- all-case fold results;
- mean, SD, and 95% confidence intervals;
- paired proposed-vs-benchmark tests;
- clean-to-artifact robustness comparisons;
- mean ± SD bar graph;
- fold-wise Accuracy and F1 graphs;
- TMD accuracy broken down by synthetic artifact category;
- 10-bin Expected Calibration Error (ECE).

The analyzer defaults to the paper's prespecified one-tailed superiority alternative (`greater`). Use `--alternative two-sided` only if the final methodology is changed before examining results. With only five paired folds, Shapiro-Wilk and hypothesis tests have low power. Report confidence intervals, fold-level values, and effect sizes alongside p-values. Cross-validation folds are also not fully independent estimates.

## 5. Grad-CAM and optional ROI localization

After final training, generate a qualitative explanation from a final checkpoint:

```bash
python generate_gradcam.py \
  --checkpoint chapter4_results/proposed_artifact_mix/fold_1_proposed_artifact_mix_best.keras \
  --image data_5_fold/fold_1/test/subluxation/IMAGE.jpg \
  --model_type proposed \
  --scenario artifact_mix
```

For Localization Energy and Grad-CAM/ROI IoU, an expert must annotate the condyle/glenoid-fossa ROI in **original-image pixel coordinates**:

```csv
sample_id,x_min,y_min,x_max,y_max
IMAGE.jpg,120,300,520,900
```

Then add `--roi_csv expert_rois.csv`. The default IoU heatmap threshold is 0.5 and must be prespecified and reported. Do not claim quantitative anatomical localization without independently prepared expert ROIs.

## Experimental cases

| Case | Model | Condition |
|---|---|---|
| C1 | DenseNet201 connected self-attention benchmark | Clean |
| C2 | DenseNet201 connected self-attention benchmark | Artifact mix |
| C3 | AGMTL-DenseCBAM | Clean |
| C4 | AGMTL-DenseCBAM | Artifact mix |

The artifact mix consists of none/clean, horizontal motion blur, Gaussian noise, and simulated bright streaks. These are **synthetic corruption categories**, not validated labels for real clinical acquisition artifacts.

## Reproducibility notes

- Image size: 224 × 224 RGB.
- Preprocessing: DenseNet ImageNet preprocessing to approximately `[-1, 1]`.
- Positive class: Subluxation (`1`).
- Validation/test artifact type and severity are deterministic per sample and seed.
- Training artifacts remain stochastic but seeded.
- Balanced class weighting is applied to the training TMD loss by default for both models; validation/test metrics remain unweighted.
- Checkpoint and early stopping monitor primary TMD validation loss.
- Each fold runs in a fresh process to reduce GPU-memory fragmentation.
- Pipeline throughput includes loading, preprocessing, and model inference; compare speed only on identical hardware/settings.
- Result directories and datasets are ignored by Git because they may contain large or sensitive files. Archive final CSV/config outputs separately with checksums.

## Known limitations

- Patient/study leakage cannot be ruled out without source metadata.
- Synthetic corruptions do not establish robustness to every real panoramic-radiography artifact.
- Grad-CAM localization is qualitative unless expert ROI annotations are available; ROI selection and heatmap thresholding can materially affect Localization Energy/IoU.
- External validation on an independent institution/device dataset is still needed.
