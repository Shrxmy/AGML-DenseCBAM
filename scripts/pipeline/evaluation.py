"""Learning curves, calibration, and prediction collection."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras import Model

from .data import TMJSequence


def save_learning_curves(
    history_df: pd.DataFrame,
    model_type: str,
    output_path: Path,
    title: str,
) -> None:
    # Save the fold learning curves.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = [("loss", "val_loss", "Total training objective")]
    if model_type == "proposed":
        panels.extend(
            [
                ("tmd_output_loss", "val_tmd_output_loss", "Primary TMD loss"),
                (
                    "artifact_output_loss",
                    "val_artifact_output_loss",
                    "Auxiliary artifact loss",
                ),
            ]
        )
    panels = [panel for panel in panels if panel[0] in history_df and panel[1] in history_df]
    if not panels:
        raise ValueError("History does not contain a training/validation loss pair.")

    epochs = np.arange(1, len(history_df) + 1)
    figure, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)

    for axis, (training_column, validation_column, panel_title) in zip(axes[0], panels):
        axis.plot(
            epochs,
            history_df[training_column].astype(float),
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="Training loss",
            color="#1f77b4",
        )
        axis.plot(
            epochs,
            history_df[validation_column].astype(float),
            marker="s",
            markersize=3,
            linewidth=1.8,
            label="Validation loss",
            color="#d62728",
        )
        axis.set_title(panel_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Training TMD loss uses balanced class sample weights; validation TMD loss is unweighted.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def unpack_batch(batch):
    # Ignore optional sample weights.
    if len(batch) == 3:
        inputs, targets, _ = batch
        return inputs, targets
    return batch


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    if not (len(y_true) == len(y_pred) == len(confidence)):
        raise ValueError("ECE arrays must have equal length.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (y_true == y_pred).astype(np.float64)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def collect_predictions(
    model: Model,
    generator: TMJSequence,
    multi_task: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []
    artifact_true: List[int] = []
    artifact_pred: List[int] = []
    artifact_conf: List[float] = []
    filepaths: List[str] = []

    for i in range(len(generator)):
        start = i * generator.batch_size
        stop = min((i + 1) * generator.batch_size, len(generator.df))
        batch_indices = generator.indices[start:stop]
        batch_df = generator.df.iloc[batch_indices]
        batch_paths = batch_df["filepath"].tolist()
        x, y = unpack_batch(generator[i])
        pred = model(x, training=False)
        if multi_task:
            pred_tmd = np.asarray(pred["tmd_output"])
            pred_artifact = np.asarray(pred["artifact_output"])
            true_tmd = y["tmd_output"]
            true_artifact = y["artifact_output"]
            artifact_true.extend(np.argmax(true_artifact, axis=1).tolist())
            artifact_pred.extend(np.argmax(pred_artifact, axis=1).tolist())
            artifact_conf.extend(np.max(pred_artifact, axis=1).tolist())
        else:
            pred_tmd = np.asarray(pred)
            true_tmd = y
            artifact_true.extend(
                generator.evaluation_artifact_label(filepath) for filepath in batch_paths
            )

        y_true.extend(np.argmax(true_tmd, axis=1).tolist())
        y_pred.extend(np.argmax(pred_tmd, axis=1).tolist())
        y_conf.extend(np.max(pred_tmd, axis=1).tolist())
        filepaths.extend(batch_paths)

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_conf),
        np.asarray(artifact_true),
        np.asarray(artifact_pred),
        np.asarray(artifact_conf),
        filepaths,
    )
