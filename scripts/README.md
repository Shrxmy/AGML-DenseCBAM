# Active scripts

- `make_5fold_dataset.py` — audits and generates folds.
- `audit_artifact_calibration.py` — produces training-only artifact previews and metrics.
- `train_one_case_5fold.py` — defines models and trains/evaluates one case; V3 is enabled explicitly with `--two_stage_fine_tuning --conservative_augmentation`.
- `run_case_5fold_isolated.py` — launches each fold in a fresh process and records the complete V2/V3 configuration fingerprint.
- `analyze_chapter4.py` — aggregates C1–C4 results and paired analyses.
- `generate_gradcam.py` — creates single-image Grad-CAM raw outputs, appendix panel, metadata, and optional ROI metrics.
- `generate_gradcam_appendix.py` — deterministically selects balanced held-out examples and creates paired benchmark/proposed appendix panels.
- `generate_learning_curves.py` — regenerates appendix-ready training/validation loss figures from existing fold histories.
- `check_tf_gpu.py` — verifies TensorFlow GPU availability.
- `run_v3_development_smoke.sh` — launches the abbreviated Fold 1 V3 validation-only functionality smoke in WSL.
- `run_v3_development_full_schedule.sh` — runs the locked 5+45-epoch V3 schedule on Fold 1 validation data without touching the outer test split.

Invoke commands from the repository root, for example:

```bash
python scripts/check_tf_gpu.py
```

For the V3 development-only Fold 1 smoke test, use the command locked in [`docs/TRAINING_V3_PROTOCOL.md`](../docs/TRAINING_V3_PROTOCOL.md). Its `--evaluation_split validation` option intentionally prevents outer-test evaluation during development.
