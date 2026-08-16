# Active scripts

- `make_5fold_dataset.py` — audits and generates folds.
- `audit_artifact_calibration.py` — produces training-only artifact previews and metrics.
- `train_one_case_5fold.py` — defines models and trains/evaluates one case.
- `run_case_5fold_isolated.py` — launches each fold in a fresh process.
- `analyze_chapter4.py` — aggregates C1–C4 results and paired analyses.
- `generate_gradcam.py` — creates Grad-CAM outputs and optional ROI metrics.
- `check_tf_gpu.py` — verifies TensorFlow GPU availability.

Invoke commands from the repository root, for example:

```bash
python scripts/check_tf_gpu.py
```
