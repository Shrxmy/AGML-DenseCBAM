# Validation-only development runner

This runner screens model changes on validation data. It produced the V3 and V4 screens. Its scores are development results and are not generalization estimates.

## What it offers

- `--attention cbam|none` and `--auxiliary-head on|off` for the proposed model
- `--recipe v2|v3` for the training recipe
- `--candidate multiscale_gated|densenet121|artifact_gated` for V4 candidates A, B and C, which need CBAM and the auxiliary head on
- `--plan-only` to print the planned commands without training

See [the guide](../../docs/GUIDE.md) for an example command.

## Guards

- **Test images:** they are only read as raw bytes for the integrity audit. They never reach a generator, model or metric.
- **Output folder:** it must be a new folder under `results/development_controlled/`. There is no overwrite or resume, so a failed run needs a new name and its folder stays as evidence.
- **Integrity audit:** before building a model, it checks manifests, file hashes, labels and exact-file leakage across splits.
- **Checkpoint selection:** all variants use the same unweighted validation cross-entropy for the diagnosis output, excluding L2 and auxiliary loss. This differs from the V2 and V3 final runs, whose monitors differ between models.
- **Provenance:** it records a source fingerprint, and a run is marked failed if the source changes while it runs.

## Output

Each run writes `run_config.json`, `provenance.json`, `run_status.json` and `validation_summary.json`. Each fold folder holds `best.weights.h5`, `checkpoint_selection.json`, `class_weights.json`, `history.csv`, `validation_predictions.csv`, `validation_confusion_matrix.csv` and `validation_metrics.json`.
