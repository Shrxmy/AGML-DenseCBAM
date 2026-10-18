# Controlled development V1 (separate from historical runs)

This proposed-only runner supports `--attention cbam|none` ×
`--auxiliary-head on|off`. It does **not** modify the historical trainer,
benchmark, Final V2, paper, folds, or existing outputs. Results are exploratory
validation results, **not** held-out generalization estimates. Validation is
reused for checkpoint selection and final prediction collection. This does not
resolve historical test exposure or establish patient/examination independence.

## Safe planning (no training)

From the repository root in WSL:

```bash
python -B scripts/run_controlled_development.py \
  --folds-root data_5_fold \
  --output-dir results/development_controlled/smoke_cbam_aux_on \
  --scenario clean --attention cbam --auxiliary-head on \
  --single-fold fold_1 --epochs 2 --plan-only
```

Plan-only prints deterministic JSON to stdout with parsed configuration, runner
version, source/config fingerprints, evaluation boundary, fold/manifest paths,
and argv arrays for all four variants. It imports no TensorFlow/scientific
packages, checks no GPU, creates no output, reads no manifest contents, enumerates
no images, and reads no image/weight files. It checks directory/manifest path
metadata only; manifest contents and split cardinalities are **not validated**
until actual execution. Source-code files are read for fingerprints. Commands
in the JSON are proposals, not executed. Every emitted sibling destination is
checked for absence and safety; each real invocation checks again.

## Real execution — operator authorization required

Removing `--plan-only` from the command above starts actual training. **Do not do
that merely to inspect the runner.** Execution imports TensorFlow, configures the
historical precision/determinism settings (including GPU discovery), may download
ImageNet weights, trains, and performs validation inference. No training or
inference was authorized or performed during implementation.

Output must be a previously **absent strict child** of the repository's
`results/development_controlled/` sandbox (e.g. `smoke_*` or a new candidate
folder). The sandbox itself, existing empty directories, symlink destinations,
and overlap with fold input or protected evidence/data/source trees are refused.
There is no overwrite/resume mode. An atomic leaf `mkdir` reserves the run before
TensorFlow imports, data auditing, weight construction, or training; a
`RUN_RESERVED` marker identifies it. Failed/partial directories are not reusable:
choose another new name. These guards assume a trusted local filesystem, not an
adversary replacing ancestors concurrently.

All selected folds are integrity-audited before building any model. The separate
development audit binds manifest paths/classes/splits/full SHA-256s to the actual
indexed files, rejects group/content cross-split leakage and conflicting labels,
and rejects symlinked image paths before reading their bytes. It **reads raw bytes
of test images for integrity auditing only**, not decoding or evaluation. Every
split must contain both classes. Counts come from actual indexed rows, not an
assumed equal split size or exact 70/15/15 ratio. This is encoded-byte identity,
not patient/near-duplicate independence. The historical pipeline is unchanged. Test rows are discarded before fold training;
they are never passed to a generator, model, prediction collector, or metric.
Prediction collection checks validation-only provenance and exact path order/count
again after restoring the selected checkpoint.

## Architecture and defaults

- Trainable ImageNet DenseNet201, `conv5_block32_concat` features.
- Optional CBAM; the primary head is always GAP → Dense1024 (ReLU, L2) →
  dropout .5 → BN → Dense128 (ReLU) → float32 softmax2, named `tmd_output`.
- When on, the unchanged pre-CBAM auxiliary head uses GAP+GMP → Dense256
  (ReLU, L2) → dropout .3 → float32 softmax4. Clean auxiliary targets are all none.
- Auxiliary-off uses an array output/target and balanced primary sample weights;
  auxiliary-on uses the existing named target/sample-weight dictionaries. The
  existing `TMJSequence` preprocessing, deterministic artifact protocol, and
  training-only horizontal flips are reused across all variants.
- Historical defaults: 224×224, batch8, maximum50 epochs, Adam1e-4, L2 .01,
  primary/auxiliary coefficients 1.0/.3, seed42, balanced training weights,
  mixed precision when GPU is available, default scenario `artifact_mix`.
- Square input >=32 is accepted; proposed-only features do not require the
  benchmark's pool-alignment restrictions. No staged fine-tuning or resolution
  benchmark has been added.

All four development cells select checkpoints using the same **unweighted
per-example primary categorical-crossentropy metric**, excluding regularization
and auxiliary loss. Output-specific Keras metric keys are normalized to
`val_primary_ce` before checkpoint, stopping, and LR callbacks consume them.
There is no fallback to total loss. This development-only selection policy fixes
the historical single/multi-output monitor asymmetry without changing FinalV2.
Actual installed-version Keras log naming remains part of the operator smoke check. Early stopping uses patience5 with best-weight
restoration; LR reduction uses factor .1/patience3/min1e-6. The runner additionally
reloads the saved best **weights-only** checkpoint before validation predictions,
even if training reaches its maximum epoch count. Missing/nonfinite monitors fail
explicitly rather than silently choosing weights.

## Records and checks

Run-level files: `RUN_RESERVED`, `run_config.json`, `provenance.json`,
`run_status.json`, `validation_summary.json`. Provenance includes the unchanged
legacy pipeline fingerprint and an authoritative `development_source_sha256`
covering this runner + development package + existing pipeline, fold-manifest
hashes, split/class counts, protocol identifiers, and environment versions.
A common-protocol hash excludes only output location and the two intended cell
toggles/effective auxiliary coefficient. Completion checks source stability;
source changes leave results as evidence but mark the run failed.
Keras state and Keras/Python/NumPy/TF seeds reset before each fold. This is
seed-paired initialization, not proof of identical random shared-head weights
across attention variants or bitwise GPU determinism. The
existing fingerprint describes current modular code, not a claim that it equals
the archived Final V2 trainer revision.
Per fold: `best.weights.h5`, `checkpoint_selection.json` (path, monitor, best epoch,
value, hash, restored status), `class_weights.json`, `history.csv`,
`validation_predictions.csv`, `validation_confusion_matrix.csv`,
`validation_metrics.json`. Status is running/completed/failed; failures return
nonzero (interrupts 130). A hard kill/power failure may leave running status; the
reservation still prevents reuse. Partial checkpoints remain as evidence.

Run only the pure/static tests without importing TensorFlow:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s tests -p test_controlled_development.py -v
```

The TF model-structure test is unconditionally skipped and marked **UNRUN** until
separate parent authorization; it permits `weights=None` only. The final full software suite ran 173 tests: 172 passed and this model-structure
check remained skipped. Existing regression tests imported TensorFlow and used
temporary synthetic image fixtures but did not construct/train/infer a model.
End-to-end training/inference checks remain deferred. The operator must launch a
fresh-directory smoke run before relying on execution compatibility with the
installed TensorFlow/Keras stack. Formatter/linter executables were unavailable;
no dependencies were installed.
