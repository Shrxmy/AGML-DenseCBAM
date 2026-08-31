# Guide

This guide covers environment setup, data preparation, V2 reproduction, statistical analysis, learning curves, and Grad-CAM generation. Run all commands from the repository root in WSL.

## 1. Environment setup

The verified environment is:

```text
/home/solyvie/environments/miniforge3/envs/thesis-env
```

Activate it:

```bash
source /home/solyvie/environments/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env
cd /home/solyvie/workspace/thesis-projects/working-run/AGML-DenseCBAM
```

For a new environment, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify TensorFlow and the GPU:

```bash
python scripts/check_tf_gpu.py
```

## 2. Run repository checks

```bash
python -m py_compile scripts/*.py
python -m unittest discover -s tests -v
```

Use these checks after changing the pipeline. Model construction can also be verified without running full training by using a one-fold smoke test as described below.

## 3. Prepare the source data

Place the source image collection under `data/`. The fold generator scans supported image files recursively and derives the Normal/Subluxation label from the class directory names.

Do not edit source images to resolve duplicates or labels. The audit and fold-generation process leaves `data/` unchanged.

### 3.1 Audit with strict conflict handling

Run the safe default first:

```bash
python scripts/make_5fold_dataset.py \
  --input_root data \
  --output_root data_5_fold \
  --n_splits 5 \
  --val_size 0.15 \
  --seed 42 \
  --conflict_policy error
```

The local dataset is expected to stop with a contradictory-label error. Review `data_5_fold/duplicate_audit.csv` before continuing.

### 3.2 Generate the approved fold set

Use exclusion only after the decision has been formally approved and documented:

```bash
python scripts/make_5fold_dataset.py \
  --input_root data \
  --output_root data_5_fold \
  --n_splits 5 \
  --val_size 0.15 \
  --seed 42 \
  --conflict_policy exclude
```

When valid patient or original-study identifiers become available, provide a CSV containing `source_path,group_id`:

```bash
python scripts/make_5fold_dataset.py \
  --input_root data \
  --output_root data_5_fold \
  --groups_csv patient_groups.csv \
  --conflict_policy exclude
```

Do not claim patient-level isolation when this metadata is absent.

## 4. Review the V2 artifact protocol

Generate training-only calibration previews and numerical summaries:

```bash
python scripts/audit_artifact_calibration.py \
  --source_root data_5_fold/fold_1/train \
  --output_dir results/artifact_calibration_v2 \
  --preview_samples 8 \
  --metric_samples 200 \
  --seed 42
```

The calibration script rejects a source path containing `test`. Do not use held-out data to choose corruption settings.

## 5. Run a smoke test

A one-epoch, one-fold run verifies data loading, model construction, training, checkpointing, held-out evaluation, and learning-curve generation:

```bash
python scripts/train_one_case_5fold.py \
  --folds_root data_5_fold \
  --output_dir results/smoke_v2/proposed_artifact_mix \
  --model_type proposed \
  --scenario artifact_mix \
  --epochs 1 \
  --batch_size 8 \
  --freeze_backbone \
  --single_fold fold_1
```

A smoke-test score is not a thesis result.

## 6. Reproduce the four final cases

Use the isolated runner so every fold starts in a fresh process. Write reproductions to a new directory rather than overwriting `results/final_v2/`.

```bash
python scripts/run_case_5fold_isolated.py \
  --model_type benchmark \
  --scenario clean \
  --epochs 50 \
  --output_dir results/reproduction_v2/benchmark_clean

python scripts/run_case_5fold_isolated.py \
  --model_type benchmark \
  --scenario artifact_mix \
  --epochs 50 \
  --output_dir results/reproduction_v2/benchmark_artifact_mix

python scripts/run_case_5fold_isolated.py \
  --model_type proposed \
  --scenario clean \
  --epochs 50 \
  --output_dir results/reproduction_v2/proposed_clean

python scripts/run_case_5fold_isolated.py \
  --model_type proposed \
  --scenario artifact_mix \
  --epochs 50 \
  --output_dir results/reproduction_v2/proposed_artifact_mix
```

Defaults reproduce the locked V2 image size, batch size, learning rate, L2 strength, seed, class weighting, and mixed-precision policy. Do not use `--skip_integrity_check` for a reportable run.

`--skip_existing` is safe only when the runner confirms that saved configuration and script fingerprints match the requested run.

## 7. Analyze the four cases

The analyzer expects these directories under its result root:

```text
benchmark_clean/
benchmark_artifact_mix/
proposed_clean/
proposed_artifact_mix/
```

Analyze the preserved final V2 outputs:

```bash
python scripts/analyze_chapter4.py \
  --results_root results/final_v2 \
  --output_dir results/regenerated/final_v2_analysis \
  --expected_folds 5
```

Analyze a new reproduction:

```bash
python scripts/analyze_chapter4.py \
  --results_root results/reproduction_v2 \
  --expected_folds 5
```

The default statistical alternative is the prespecified one-tailed `greater` hypothesis. Use a two-sided test only when required by the approved analysis plan:

```bash
python scripts/analyze_chapter4.py \
  --results_root results/reproduction_v2 \
  --expected_folds 5 \
  --alternative two-sided
```

Do not replace the prespecified analysis after viewing results without documenting the change.

## 8. Regenerate learning curves

Learning curves are created from saved history CSV files and do not require retraining:

```bash
python scripts/generate_learning_curves.py \
  --results_root results/final_v2 \
  --output_root results/regenerated/final_v2_learning_curves
```

The separate output root preserves the checksummed Final V2 figures. Each generated case directory should contain five files named `fold_N_learning_curves.png`. Proposed-model figures show total objective, TMD loss, and auxiliary artifact loss. Training TMD loss is weighted while validation TMD loss is unweighted.

## 9. Generate the Grad-CAM appendix

Generate one balanced sample per true class for each fold and scenario:

```bash
python scripts/generate_gradcam_appendix.py \
  --results_root results/final_v2 \
  --output_dir results/regenerated/final_v2_gradcam_appendix \
  --samples_per_class 1 \
  --seed 42
```

The expected output is:

```text
40 individual model/sample panels
20 paired benchmark/proposed comparison panels
gradcam_appendix_metadata.csv
gradcam_appendix_config.json
```

The separate output directory avoids changing the checksummed Final V2 appendix. Selection uses a stable SHA-256 rule and does not depend on model correctness, confidence, or heatmap appearance. Do not search multiple seeds to obtain more favorable visualizations.

Recommended caption:

> **Grad-CAM comparison for a deterministically selected held-out image.** The panels show the exact model input, class-targeted Grad-CAM heatmap, and overlay. Samples were selected by a prespecified stable-hash rule independent of model correctness or heatmap appearance. The visualizations are qualitative and do not establish anatomical localization accuracy without expert region-of-interest annotations.

## 10. Generate Grad-CAM for one image

```bash
python scripts/generate_gradcam.py \
  --checkpoint results/final_v2/proposed_artifact_mix/fold_1_proposed_artifact_mix_best.keras \
  --image data_5_fold/fold_1/test/normal/<FILENAME>.jpg \
  --model_type proposed \
  --scenario artifact_mix \
  --precision_policy auto \
  --output_dir results/gradcam/manual_case
```

`auto` restores the precision policy from the checkpoint's adjacent `run_config.json`. For a legacy checkpoint without configuration metadata, explicitly use `float32` or `mixed_float16`.

The command saves the exact model input, grayscale heatmap, color overlay, panel figure, and JSON provenance metadata.

### Optional expert ROI metrics

The ROI CSV format is:

```csv
sample_id,x_min,y_min,x_max,y_max
<filename>.jpg,100,80,420,360
```

Use it only when annotations were prepared independently by a qualified expert:

```bash
python scripts/generate_gradcam.py \
  --checkpoint <CHECKPOINT.keras> \
  --image <IMAGE.jpg> \
  --model_type proposed \
  --scenario artifact_mix \
  --roi_csv expert_rois.csv \
  --heatmap_threshold 0.5 \
  --output_dir results/gradcam/with_roi
```

Document the annotator, procedure, coordinate system, and prespecified threshold. Annotations created after viewing model heatmaps introduce confirmation bias.

## 11. Use the notebook interface

Launch Jupyter from the repository root:

```bash
jupyter lab notebooks/AGML-DenseCBAM-Training.ipynb
```

Select the `thesis-env` kernel. The notebook is a thin interface; active model and analysis logic remains under `scripts/`. Its default output is `results/reproduction_v2/`, and training and analysis cells are disabled until explicitly enabled.

## 12. Verify preserved checksums

From the repository root:

```bash
sha256sum --check results/final_3002_seed42.sha256
sha256sum --check results/final_v2.sha256
```

A missing file or mismatch must be investigated before using preserved outputs in the thesis.

## 13. Output structure

A completed result root follows this layout:

```text
results/<run_name>/
├── benchmark_clean/
├── benchmark_artifact_mix/
├── proposed_clean/
├── proposed_artifact_mix/
├── analysis/
└── gradcam_appendix/
```

Each case contains fold checkpoints, predictions, confusion matrices, metric rows, histories, learning curves, aggregate tables, and a run configuration.

## 14. Common problems

### GPU is not detected

Run `python scripts/check_tf_gpu.py`. Confirm that WSL can access the NVIDIA driver and that the pinned TensorFlow CUDA dependencies are installed in the active environment.

### Fold generation stops on conflicting labels

This is expected for the audited local dataset. Review `duplicate_audit.csv`. Do not bypass the error until exclusion or correction is approved and documented.

### Existing results are rejected

The requested configuration, fold manifests, or active scripts differ from the saved fingerprints. Use a new output directory rather than altering or mixing incompatible fold outputs.

### GPU memory is insufficient

Reduce batch size only for development. A changed batch size is a protocol deviation and must not be mixed with the locked final comparison.

### Grad-CAM cannot relocate an archived path

Run from the repository root or provide `--project_root` to the appendix generator. The held-out image must still resolve inside the expected test fold.

## 15. Reporting safeguards

Before treating outputs as thesis evidence, confirm that:

1. the contradictory-label decision is formally documented;
2. all four cases use identical folds and matched settings;
3. the held-out test data did not influence training or checkpoint selection;
4. no case output contains NaN or infinite values;
5. all expected fold files and fingerprints pass validation;
6. efficiency measurements came from the same hardware environment;
7. synthetic artifacts are described as controlled stress tests; and
8. conclusions match the qualified statistical findings in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).
