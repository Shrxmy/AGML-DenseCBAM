"""Fold training, held-out evaluation, and result-file generation."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import mixed_precision

from .config import (
    ARTIFACT_LABELS,
    ARTIFACT_PROTOCOL,
    DISPLAY_LABELS,
    TRAINING_PROTOCOL,
    RunConfig,
    configure_tensorflow,
    effective_run_config_sha256,
    seed_everything,
)
from .data import TMJSequence, balanced_class_weights, index_split_dataset
from .evaluation import (
    collect_predictions,
    expected_calibration_error,
    save_learning_curves,
    unpack_batch,
)
from .models import build_benchmark_model, build_proposed_model, compile_model, fit_model
from .provenance import sha256_file, training_source_sha256


def run_one_fold(fold_root: Path, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Running {config.model_type} | {config.scenario} | {fold_root.name} ===")
    seed_everything(config.random_state)

    df = index_split_dataset(fold_root, verify_integrity=config.verify_integrity)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    evaluation_df = test_df
    print(df.groupby(["split", "class_name"]).size())
    print(f"Evaluation split: test ({len(evaluation_df)} images)")

    multi_task = config.model_type == "proposed"
    class_weights = balanced_class_weights(train_df["tmd_label"]) if config.class_weighting else None
    print("TMD class weights:", class_weights)
    train_gen = TMJSequence(
        train_df,
        config.image_size,
        config.batch_size,
        multi_task,
        config.scenario,
        True,
        config.random_state,
        tmd_class_weights=class_weights,
    )
    val_gen = TMJSequence(
        val_df,
        config.image_size,
        config.batch_size,
        multi_task,
        config.scenario,
        False,
        config.random_state,
    )
    evaluation_gen = TMJSequence(
        evaluation_df,
        config.image_size,
        config.batch_size,
        multi_task,
        config.scenario,
        False,
        config.random_state,
    )

    model = build_proposed_model(config) if multi_task else build_benchmark_model(config)
    compile_model(model, config)

    case_dir = config.output_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = case_dir / f"{fold_root.name}_{config.model_type}_{config.scenario}_best.keras"
    history_df = fit_model(
        model,
        train_gen,
        val_gen,
        config,
        checkpoint_path,
    )
    learning_curve_path = case_dir / f"{fold_root.name}_learning_curves.png"
    save_learning_curves(
        history_df,
        config.model_type,
        learning_curve_path,
        title=f"{fold_root.name.replace('_', ' ').title()} — {config.model_type.title()} / {config.scenario}",
    )

    # Warm up before timing.
    warmup_x, _ = unpack_batch(evaluation_gen[0])
    _ = model(warmup_x, training=False)
    start = time.perf_counter()
    (
        y_true,
        y_pred,
        y_conf,
        artifact_true,
        artifact_pred,
        artifact_conf,
        prediction_paths,
    ) = collect_predictions(model, evaluation_gen, multi_task)
    elapsed = time.perf_counter() - start
    images_per_second = len(y_true) / max(elapsed, 1e-8)
    latency_ms = 1000.0 / max(images_per_second, 1e-8)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    manifest_path = fold_root / "manifest.csv"
    fold_manifest_sha256 = sha256_file(manifest_path) if manifest_path.exists() else None

    per_artifact_metrics: Dict[str, float | int] = {}
    for artifact_index, artifact_name in enumerate(ARTIFACT_LABELS):
        mask = artifact_true == artifact_index
        per_artifact_metrics[f"n_{artifact_name}"] = int(mask.sum())
        per_artifact_metrics[f"tmd_accuracy_{artifact_name}"] = (
            accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else np.nan
        )
        per_artifact_metrics[f"artifact_recall_{artifact_name}"] = (
            float(np.mean(artifact_pred[mask] == artifact_index))
            if multi_task and mask.any()
            else np.nan
        )

    result_df = pd.DataFrame(
        [
            {
                "model_type": config.model_type,
                "scenario": config.scenario,
                "fold": fold_root.name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "specificity": tn / max(tn + fp, 1),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "artifact_accuracy": (
                    accuracy_score(artifact_true, artifact_pred) if multi_task else np.nan
                ),
                "artifact_macro_f1": (
                    f1_score(
                        artifact_true,
                        artifact_pred,
                        labels=list(range(len(ARTIFACT_LABELS))),
                        average="macro",
                        zero_division=0,
                    )
                    if multi_task
                    else np.nan
                ),
                "ece_10_bins": expected_calibration_error(y_true, y_pred, y_conf, n_bins=10),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "images_per_second": images_per_second,
                "latency_ms": latency_ms,
                "learning_curve_png": str(learning_curve_path),
                "epochs_ran": len(history_df),
                "evaluation_split": "test",
                "training_protocol": TRAINING_PROTOCOL,
                "tensorflow_version": tf.__version__,
                "precision_policy": mixed_precision.global_policy().name,
                "artifact_protocol": ARTIFACT_PROTOCOL,
                "class_weighting": config.class_weighting,
                "fold_manifest_sha256": fold_manifest_sha256,
                "training_script_sha256": training_source_sha256(),
                "run_config_sha256": effective_run_config_sha256(config),
                **per_artifact_metrics,
            }
        ]
    )
    cm_df = pd.DataFrame(cm, index=DISPLAY_LABELS, columns=DISPLAY_LABELS)
    pred_data: Dict[str, object] = {
        "filepath": prediction_paths,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": y_conf,
        "artifact_true": artifact_true,
    }
    if multi_task:
        pred_data.update(
            {
                "artifact_pred": artifact_pred,
                "artifact_confidence": artifact_conf,
            }
        )
    pred_df = pd.DataFrame(pred_data)

    result_df.to_csv(case_dir / f"{fold_root.name}_results.csv", index=False)
    cm_df.to_csv(case_dir / f"{fold_root.name}_confusion_matrix.csv")
    if multi_task:
        artifact_cm = confusion_matrix(
            artifact_true,
            artifact_pred,
            labels=list(range(len(ARTIFACT_LABELS))),
        )
        pd.DataFrame(
            artifact_cm,
            index=ARTIFACT_LABELS,
            columns=ARTIFACT_LABELS,
        ).to_csv(case_dir / f"{fold_root.name}_artifact_confusion_matrix.csv")
    pred_df.to_csv(case_dir / f"{fold_root.name}_predictions.csv", index=False)
    history_df.to_csv(case_dir / f"{fold_root.name}_history.csv", index=False)

    print(result_df)
    print(cm_df)

    tf.keras.backend.clear_session()
    gc.collect()
    return result_df, cm_df, pred_df, history_df


def run_case(config: RunConfig) -> pd.DataFrame:
    configure_tensorflow(config)
    if config.model_type not in {"benchmark", "proposed"}:
        raise ValueError("model_type must be 'benchmark' or 'proposed'")
    if config.scenario not in {"clean", "artifact_mix"}:
        raise ValueError("scenario must be 'clean' or 'artifact_mix'")

    fold_dirs = sorted(
        [p for p in config.folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if config.single_fold is not None:
        fold_dirs = [p for p in fold_dirs if p.name == config.single_fold]
        if not fold_dirs:
            raise FileNotFoundError(
                f"Requested --single_fold {config.single_fold!r}, but it was not found under {config.folds_root}"
            )
    if config.fold_limit is not None:
        fold_dirs = fold_dirs[: config.fold_limit]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold folders found under {config.folds_root}")

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Config:", config)

    result_rows: List[pd.DataFrame] = []
    pooled_cm: pd.DataFrame | None = None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        serializable = asdict(config)
        serializable["folds_root"] = str(config.folds_root)
        serializable["output_dir"] = str(config.output_dir)
        serializable["tensorflow_version"] = tf.__version__
        serializable["gpu_devices"] = [device.name for device in tf.config.list_physical_devices("GPU")]
        serializable["precision_policy"] = mixed_precision.global_policy().name
        serializable["artifact_protocol"] = ARTIFACT_PROTOCOL
        serializable["training_protocol"] = TRAINING_PROTOCOL
        serializable["training_script_sha256"] = training_source_sha256()
        serializable["effective_run_config_sha256"] = effective_run_config_sha256(config)
        json.dump(serializable, f, indent=2)

    for fold_root in fold_dirs:
        result_df, cm_df, _, _ = run_one_fold(fold_root, config)
        result_rows.append(result_df)
        pooled_cm = cm_df.copy() if pooled_cm is None else pooled_cm.add(cm_df, fill_value=0)

    combined = pd.concat(result_rows, ignore_index=True)
    combined.to_csv(config.output_dir / "all_fold_results.csv", index=False)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "ece_10_bins",
        "images_per_second",
        "latency_ms",
        "epochs_ran",
    ]
    if combined["artifact_accuracy"].notna().any():
        metric_columns.extend(["artifact_accuracy", "artifact_macro_f1"])
        metric_columns.extend(
            f"artifact_recall_{artifact_name}" for artifact_name in ARTIFACT_LABELS
        )
    summary = combined[metric_columns].agg(["mean", "std"]).T.reset_index(names="metric")
    summary.to_csv(config.output_dir / "summary_mean_std.csv", index=False)

    if pooled_cm is not None:
        pooled_cm.astype(int).to_csv(config.output_dir / "pooled_confusion_matrix.csv")

    print("\n=== Combined fold results ===")
    print(combined)
    print("\n=== Mean ± SD summary ===")
    print(summary)
    return combined
