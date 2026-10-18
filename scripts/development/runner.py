"""Execution boundary: train/validation generators only; test bytes audit only.

Importing this module is safe without TensorFlow. Actual execution is a separate
user-launched operation, and requires an atomically reserved fresh output first.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path

from scripts.pipeline.provenance import sha256_file, sha256_json

from .config import (
    EVALUATION_BOUNDARY, PROJECT_ROOT, DevelopmentConfig, discover_folds,
    reserve_output, source_provenance, validate_config, validate_index_rows,
    )


def write_json(path: Path, payload: dict) -> None:
    """Atomic metadata replacement, only within the newly reserved output."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def environment_versions() -> dict:
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ("tensorflow", "tensorflow-cpu", "keras", "numpy", "pandas", "opencv-python", "opencv-python-headless", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def select_checkpoint(values) -> tuple[int, float]:
    """First strict minimum, matching save_best_only and EarlyStopping ties."""
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Checkpoint monitor must have nonempty finite history")
    index = min(range(len(values)), key=values.__getitem__)
    return index + 1, values[index]


def normalize_primary_monitor(logs: dict) -> float:
    """Normalize only the unweighted CE metric; never fall back to total loss."""
    keys = [key for key in ("val_primary_ce", "val_tmd_output_primary_ce") if key in logs]
    if not keys:
        raise RuntimeError(f"Missing primary CE validation metric; available logs: {sorted(logs)}")
    values = [float(logs[key]) for key in keys]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("Nonfinite primary CE validation metric")
    if len(values) == 2 and values[0] != values[1]:
        raise RuntimeError("Conflicting primary CE validation metric aliases")
    logs["val_primary_ce"] = values[0]
    return values[0]


def common_protocol_payload(config: DevelopmentConfig) -> dict:
    payload = config.payload()
    for key in ("output_dir", "attention", "auxiliary_head", "effective_artifact_loss_weight"):
        payload.pop(key, None)
    return payload


def make_sequences(config, train_df, validation_df, class_weights, sequence_factory):
    """The only generator factory; reject wrong/empty partitions before loading."""
    for frame, split in ((train_df, "train"), (validation_df, "validation")):
        if len(frame) == 0 or set(frame["split"]) != {split}:
            raise ValueError(f"Generator input must contain only nonempty {split} rows")
    common = dict(image_size=config.input_hw, batch_size=config.batch_size,
                  multi_task=config.multi_task, scenario=config.scenario, seed=config.seed, recipe=config.recipe)
    train = sequence_factory(train_df, training=True, tmd_class_weights=class_weights, **common)
    validation = sequence_factory(validation_df, training=False, **common)
    return train, validation


def collect_validation_predictions(model, generator, multi_task, collector):
    """Check the boundary again at post-checkpoint prediction collection."""
    if generator.training or len(generator.df) == 0 or set(generator.df["split"]) != {"validation"}:
        raise ValueError("Prediction collection requires a nontraining validation-only generator")
    result = collector(model, generator, multi_task)
    paths = result[-1]
    expected = generator.df["filepath"].tolist()
    if len(paths) != len(expected) or list(paths) != expected:
        raise ValueError("Validation prediction paths/cardinality/order do not match the indexed validation split")
    return result


def pipeline_config(config: DevelopmentConfig):
    # Stable RunConfig imports TF, so never use it in config/planning code.
    from scripts.pipeline.config import RunConfig

    return RunConfig(
        folds_root=config.folds_root.resolve(), output_dir=config.output_dir.resolve(),
        model_type="proposed", scenario=config.scenario,
        image_size=config.input_hw, batch_size=config.batch_size,
        epochs=config.epochs, learning_rate=config.learning_rate, l2_strength=config.l2_strength,
        random_state=config.seed, freeze_backbone=False, tmd_loss_weight=1.0,
        artifact_loss_weight=config.artifact_loss_weight, fold_limit=None,
        single_fold=config.single_fold, verify_integrity=True, mixed_precision=True,
        class_weighting=True, run_config_sha256=sha256_json(config.payload()), recipe=config.recipe,
    )


def audit_indexed_fold(fold: Path):
    """Strict one-snapshot manifest binding; no image decoder or model execution."""
    from .integrity import audit_fold
    rows, audit = audit_fold(fold)
    import pandas as pd
    return pd.DataFrame(rows), audit


def run_fold(config: DevelopmentConfig, fold: Path, train_df, validation_df, output: Path) -> dict:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from scripts.pipeline.config import seed_everything
    from scripts.pipeline.data import TMJSequence, balanced_class_weights
    from scripts.pipeline.evaluation import collect_predictions, expected_calibration_error
    from .models import build_model, compile_model

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    seed_everything(config.seed)
    fold_output = output / fold.name
    fold_output.mkdir(exist_ok=False)
    checkpoint = fold_output / "best.weights.h5"
    monitor = config.monitor
    selection = {"monitor": monitor, "mode": "min", "checkpoint_path": str(checkpoint),
                 "checkpoint_selection_split": "validation", "best_epoch": None,
                 "best_monitored_value": None, "restored_best_weights": False,
                 "objective_note": config.payload()["checkpoint_objective_note"]}
    write_json(fold_output / "checkpoint_selection.json", selection)
    class_weights = balanced_class_weights(train_df["tmd_label"])
    write_json(fold_output / "class_weights.json", {str(key): float(value) for key, value in class_weights.items()})
    train_gen, validation_gen = make_sequences(config, train_df, validation_df, class_weights, TMJSequence)
    model = build_model(config)
    compile_model(model, config)

    class PrimaryMonitorAlias(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise RuntimeError("Missing epoch logs")
            normalize_primary_monitor(logs)

    class MonitorGuard(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"Missing/nonfinite checkpoint monitor {monitor!r}; available logs: {sorted(logs or {})}")

    class CheckpointRecord(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            value = float(logs[monitor])
            best = selection["best_monitored_value"]
            if best is None or value < best:
                if not checkpoint.is_file():
                    raise RuntimeError("Best checkpoint was not saved")
                selection.update(best_epoch=epoch + 1, best_monitored_value=value)
                write_json(fold_output / "checkpoint_selection.json", selection)

    callbacks = [
        PrimaryMonitorAlias(),
        MonitorGuard(),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=monitor, mode="min",
                                           save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="min", factor=0.1, patience=3, min_lr=1e-6, verbose=1),
        CheckpointRecord(),
    ]
    history = pd.DataFrame(model.fit(train_gen, epochs=config.epochs, validation_data=validation_gen,
                                     callbacks=callbacks, verbose=1).history)
    history.to_csv(fold_output / "history.csv", index=False)
    best_epoch, best_value = select_checkpoint(history[monitor])
    if selection["best_epoch"] != best_epoch or selection["best_monitored_value"] != best_value:
        raise RuntimeError("Checkpoint metadata disagrees with training history")
    # Explicit restore also covers reaching the configured maximum epoch count.
    model.load_weights(str(checkpoint))
    selection.update(restored_best_weights=True, epochs_ran=len(history), checkpoint_sha256=sha256_file(checkpoint))
    write_json(fold_output / "checkpoint_selection.json", selection)

    y_true, y_pred, confidence, artifact_true, artifact_pred, artifact_confidence, paths = collect_validation_predictions(
        model, validation_gen, config.multi_task, collect_predictions,
    )
    predictions = {"evaluation_split": "validation", "filepath": paths, "y_true": y_true, "y_pred": y_pred,
                   "confidence": confidence, "artifact_true": artifact_true}
    if config.multi_task:
        predictions.update(artifact_pred=artifact_pred, artifact_confidence=artifact_confidence)
    pd.DataFrame(predictions).to_csv(fold_output / "validation_predictions.csv", index=False)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(matrix, index=["Normal", "Subluxation"], columns=["Normal", "Subluxation"]).to_csv(
        fold_output / "validation_confusion_matrix.csv")
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "fold": fold.name, "evaluation_boundary": EVALUATION_BOUNDARY,
        "n_train": len(train_df), "n_validation": len(validation_df),
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "validation_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "validation_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "validation_specificity": tn / max(tn + fp, 1),
        "validation_ece_10_bins": expected_calibration_error(y_true, y_pred, confidence),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "validation_artifact_accuracy": float(np.mean(artifact_true == artifact_pred)) if config.multi_task else None,
        "validation_artifact_macro_f1": float(f1_score(artifact_true, artifact_pred, labels=[0, 1, 2, 3],
                                                       average="macro", zero_division=0)) if config.multi_task else None,
        "checkpoint_selection": selection,
    }
    write_json(fold_output / "validation_metrics.json", metrics)
    tf.keras.backend.clear_session()
    return metrics


def execute(config: DevelopmentConfig, project_root: Path = PROJECT_ROOT) -> dict:
    """No runtime dependency import, weight construction, or fit before reservation.

    All selected folds are integrity-audited before any model is built. The
    development-local audit reads/hashes test bytes for integrity binding ONLY. We never
    decode test images or hand test rows to a sequence or prediction function.
    """
    validate_config(config)
    folds = discover_folds(config)
    output = reserve_output(config, project_root)
    status = {"state": "running", "exit_code": None, "completed_folds": [], "active_fold": None,
              "evaluation_boundary": EVALUATION_BOUNDARY}
    try:
        write_json(output / "run_status.json", status)
        write_json(output / "run_config.json", config.payload())
        metadata = {"run_config_sha256": sha256_json(config.payload()), "sources": source_provenance(project_root),
                    "environment": environment_versions(), "evaluation_boundary": EVALUATION_BOUNDARY,
                    "common_protocol_sha256": sha256_json(common_protocol_payload(config)),
                    "initialization_policy": "Keras/Python/NumPy/TF seed reset per fold; seed-paired, not identical shared random head weights across attention variants.",
                    "fold_manifest_sha256": {}, "split_metadata": {}, "manifest_split_counts": {},
                    "integrity_audits": {}}
        write_json(output / "provenance.json", metadata)
        # These imports may initialize TF, but only after the output is reserved.
        from scripts.pipeline.config import ARTIFACT_PROTOCOL, TRAINING_PROTOCOL, configure_tensorflow
        
        partitions = []
        for fold in folds:
            status["active_fold"] = fold.name
            write_json(output / "run_status.json", status)
            indexed, audit = audit_indexed_fold(fold)
            counts = validate_index_rows(indexed.to_dict("records"), fold)
            if counts != audit["split_metadata"]:
                raise RuntimeError("Audited split metadata does not match indexed rows")
            metadata["fold_manifest_sha256"][fold.name] = audit["manifest_sha256"]
            metadata["split_metadata"][fold.name] = counts
            metadata["manifest_split_counts"][fold.name] = audit["manifest_split_counts"]
            metadata["integrity_audits"][fold.name] = audit
            write_json(output / "provenance.json", metadata)
            # Test rows are discarded here; run_fold receives only these two frames.
            partitions.append((fold, indexed[indexed["split"] == "train"].reset_index(drop=True),
                               indexed[indexed["split"] == "validation"].reset_index(drop=True)))
        configure_tensorflow(pipeline_config(config))
        import tensorflow as tf

        metadata.update(tensorflow_version=tf.__version__, precision_policy=tf.keras.mixed_precision.global_policy().name,
                        artifact_protocol=ARTIFACT_PROTOCOL, training_protocol=TRAINING_PROTOCOL)
        if config.recipe != "v2":
            from scripts.pipeline.config import TRAINING_PROTOCOLS
            metadata["training_protocol"] = TRAINING_PROTOCOLS[config.recipe]
        if hasattr(tf, "config"):
            devices = tf.config.list_physical_devices("GPU")
            metadata["gpu_devices"] = [str(device) for device in devices]
            deterministic_query = getattr(tf.config.experimental, "is_op_determinism_enabled", None)
            metadata["deterministic_ops_enabled"] = deterministic_query() if deterministic_query else None
        write_json(output / "provenance.json", metadata)
        results = []
        for fold, train_df, validation_df in partitions:
            status["active_fold"] = fold.name
            write_json(output / "run_status.json", status)
            results.append(run_fold(config, fold, train_df, validation_df, output))
            status["completed_folds"].append(fold.name)
            write_json(output / "run_status.json", status)
        summary = {"evaluation_boundary": EVALUATION_BOUNDARY, "fold_results": results,
                   "note": "Descriptive development validation results, reused for checkpoint selection; not confirmatory generalization."}
        write_json(output / "validation_summary.json", summary)
        final_sources = source_provenance(project_root)
        metadata["sources_unchanged_during_run"] = final_sources == metadata["sources"]
        metadata["completion_sources"] = final_sources
        write_json(output / "provenance.json", metadata)
        if not metadata["sources_unchanged_during_run"]:
            raise RuntimeError("Executed source files changed during the run; results retained but run marked failed")
        status.update(state="completed", exit_code=0, active_fold=None)
        write_json(output / "run_status.json", status)
        return summary
    except BaseException as exc:
        status.update(state="failed", exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1,
                      error_type=type(exc).__name__, error=str(exc))
        write_json(output / "run_status.json", status)
        raise
