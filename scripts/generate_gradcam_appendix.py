#!/usr/bin/env python
# Generate deterministic balanced Grad-CAM appendix panels.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .generate_gradcam import (
        TARGET_LAYERS,
        gradcam,
        load_checkpoint,
        load_input,
        overlay_heatmap,
        save_gradcam_panel,
    )
    from .train_one_case_5fold import ARTIFACT_LABELS, DISPLAY_LABELS
except ImportError:  # Direct execution
    from generate_gradcam import (
        TARGET_LAYERS,
        gradcam,
        load_checkpoint,
        load_input,
        overlay_heatmap,
        save_gradcam_panel,
    )
    from train_one_case_5fold import ARTIFACT_LABELS, DISPLAY_LABELS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(recorded_path: str, project_root: Path) -> Path:
    path = Path(recorded_path)
    if path.exists():
        return path.resolve()
    normalized_parts = Path(recorded_path.replace("\\", "/")).parts
    if "data_5_fold" in normalized_parts:
        relative = normalized_parts[normalized_parts.index("data_5_fold") :]
        candidate = project_root.joinpath(*relative)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Prediction image path is unavailable: {recorded_path}. "
        f"Tried relocating it under {project_root}."
    )


def validate_heldout_predictions(predictions: pd.DataFrame, fold: str) -> None:
    for filepath in predictions["filepath"].astype(str):
        normalized = filepath.replace("\\", "/")
        parts = normalized.split("/")
        if fold not in parts or "test" not in parts:
            raise ValueError(
                f"Grad-CAM appendix requires held-out {fold}/test images; found {filepath}"
            )


def stable_selection_key(seed: int, scenario: str, fold: str, filepath: str) -> str:
    value = f"{seed}:{scenario}:{fold}:{Path(filepath).name}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_balanced_samples(
    predictions: pd.DataFrame,
    seed: int,
    scenario: str,
    fold: str,
    samples_per_class: int,
) -> pd.DataFrame:
    required = {"filepath", "y_true"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    selected = []
    for class_index in range(len(DISPLAY_LABELS)):
        candidates = predictions[predictions["y_true"] == class_index].copy()
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"{fold}/{scenario} has only {len(candidates)} samples for class {class_index}."
            )
        candidates["selection_key"] = candidates["filepath"].map(
            lambda filepath: stable_selection_key(seed, scenario, fold, filepath)
        )
        selected.append(candidates.sort_values("selection_key").head(samples_per_class))
    return pd.concat(selected, ignore_index=True)


def save_raw_gradcam_outputs(
    output_dir: Path,
    stem: str,
    display_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    image_size: int,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{stem}_input.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(input_path), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(heatmap_path),
        np.uint8(cv2.resize(heatmap, (image_size, image_size)) * 255),
    )
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "input_png": str(input_path),
        "heatmap_png": str(heatmap_path),
        "overlay_png": str(overlay_path),
    }


def save_paired_comparison_panels(metadata: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Create one paired comparison per selected input.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = metadata.copy()
    metadata["comparison_panel_png"] = ""
    group_columns = ["scenario", "fold", "true_class", "filepath"]
    for group_values, group in metadata.groupby(group_columns, sort=True):
        scenario, fold, true_class, filepath = group_values
        if set(group["model_type"]) != {"benchmark", "proposed"}:
            raise ValueError(f"Paired Grad-CAM group is incomplete: {group_values}")
        figure, axes = plt.subplots(2, 3, figsize=(10.8, 7.0))
        for row_index, model_type in enumerate(["benchmark", "proposed"]):
            row = group[group["model_type"] == model_type].iloc[0]
            input_bgr = cv2.imread(str(row["input_png"]))
            heatmap = cv2.imread(str(row["heatmap_png"]), cv2.IMREAD_GRAYSCALE)
            overlay_bgr = cv2.imread(str(row["overlay_png"]))
            if input_bgr is None or heatmap is None or overlay_bgr is None:
                raise ValueError(f"Could not read generated Grad-CAM files for {row['filepath']}")
            axes[row_index, 0].imshow(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
            axes[row_index, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=255)
            axes[row_index, 2].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
            axes[row_index, 0].set_ylabel(
                f"{model_type.title()}\nPred: {row['predicted_class']}\n"
                f"P(Sub)={float(row['subluxation_probability']):.3f}",
                fontsize=10,
            )
        for column, title in enumerate(["Model input", "Grad-CAM heatmap", "Overlay"]):
            axes[0, column].set_title(title)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        image_stem = Path(str(filepath)).stem
        comparison_dir = output_dir / "comparisons" / str(scenario) / str(fold)
        comparison_path = comparison_dir / (
            f"{fold}_{scenario}_true-{str(true_class).lower()}_{image_stem}_comparison.png"
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        figure.suptitle(
            f"{str(fold).replace('_', ' ').title()} — {scenario} — True class: {true_class}",
            fontsize=13,
            fontweight="bold",
        )
        figure.tight_layout()
        figure.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        mask = np.ones(len(metadata), dtype=bool)
        for column, value in zip(group_columns, group_values):
            mask &= metadata[column].astype(str).to_numpy() == str(value)
        metadata.loc[mask, "comparison_panel_png"] = str(comparison_path)
    return metadata


def generate_appendix(
    results_root: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int,
    image_size: int,
    alpha: float,
    expected_folds: int,
    project_root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in ["clean", "artifact_mix"]:
        selection_case = results_root / f"proposed_{scenario}"
        run_config_path = selection_case / "run_config.json"
        if not run_config_path.exists():
            raise FileNotFoundError(f"Missing run configuration: {run_config_path}")
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        recorded_seed = int(run_config.get("random_state", -1))
        recorded_size = tuple(run_config.get("image_size", []))
        if seed != recorded_seed:
            raise ValueError(
                f"Grad-CAM seed {seed} must match the recorded evaluation seed {recorded_seed}."
            )
        if recorded_size and recorded_size != (image_size, image_size):
            raise ValueError(
                f"Grad-CAM image size {image_size} does not match recorded size {recorded_size}."
            )
        all_results_path = selection_case / "all_fold_results.csv"
        if all_results_path.exists():
            all_results = pd.read_csv(all_results_path)
            if "evaluation_split" in all_results and set(all_results["evaluation_split"].dropna()) != {"test"}:
                raise ValueError(f"{all_results_path} is not an outer-test result set.")
        fold_prediction_paths = sorted(selection_case.glob("fold_*_predictions.csv"))
        if len(fold_prediction_paths) != expected_folds:
            raise ValueError(
                f"Expected {expected_folds} prediction folds under {selection_case}; "
                f"found {len(fold_prediction_paths)}."
            )

        for prediction_path in fold_prediction_paths:
            fold = prediction_path.name.removesuffix("_predictions.csv")
            predictions = pd.read_csv(prediction_path)
            validate_heldout_predictions(predictions, fold)
            prediction_table_sha256 = sha256_file(prediction_path)
            selected = select_balanced_samples(
                predictions,
                seed,
                scenario,
                fold,
                samples_per_class,
            )
            for model_type in ["benchmark", "proposed"]:
                case_dir = results_root / f"{model_type}_{scenario}"
                checkpoint = case_dir / f"{fold}_{model_type}_{scenario}_best.keras"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                checkpoint_sha256 = sha256_file(checkpoint)
                model, precision_policy = load_checkpoint(checkpoint)

                for sample_number, sample in selected.iterrows():
                    recorded_filepath = str(sample["filepath"])
                    image_path = resolve_image_path(recorded_filepath, project_root)
                    inputs, display_rgb, artifact_label = load_input(
                        image_path,
                        model_type,
                        scenario,
                        image_size,
                        seed,
                    )
                    heatmap, probabilities, target_class = gradcam(
                        model,
                        inputs,
                        TARGET_LAYERS[model_type],
                        class_index=None,
                    )
                    overlay = overlay_heatmap(display_rgb, heatmap, alpha)
                    class_name = DISPLAY_LABELS[int(sample["y_true"])]
                    stem = (
                        f"{fold}_{model_type}_{scenario}_true-{class_name.lower()}_"
                        f"sample-{sample_number + 1}_{image_path.stem}"
                    )
                    sample_dir = output_dir / scenario / fold
                    paths = save_raw_gradcam_outputs(
                        sample_dir,
                        stem,
                        display_rgb,
                        heatmap,
                        overlay,
                        image_size,
                    )
                    panel_path = sample_dir / f"{stem}_panel.png"
                    save_gradcam_panel(
                        display_rgb,
                        heatmap,
                        overlay,
                        probabilities,
                        target_class,
                        panel_path,
                        title=f"{fold.replace('_', ' ').title()} — {model_type.title()} / {scenario}",
                    )
                    predicted_class = int(np.argmax(probabilities))
                    rows.append(
                        {
                            "fold": fold,
                            "scenario": scenario,
                            "model_type": model_type,
                            "filepath": str(image_path),
                            "recorded_filepath": recorded_filepath,
                            "source_image_sha256": sha256_file(image_path),
                            "model_input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
                            "checkpoint_sha256": checkpoint_sha256,
                            "prediction_table_sha256": prediction_table_sha256,
                            "selection_key": sample["selection_key"],
                            "selection_rule": "stable SHA-256 within fold/scenario/true class",
                            "selection_seed": seed,
                            "image_size": image_size,
                            "overlay_alpha": alpha,
                            "true_class_index": int(sample["y_true"]),
                            "true_class": class_name,
                            "predicted_class_index": predicted_class,
                            "predicted_class": DISPLAY_LABELS[predicted_class],
                            "correct": predicted_class == int(sample["y_true"]),
                            "normal_probability": float(probabilities[0]),
                            "subluxation_probability": float(probabilities[1]),
                            "gradcam_target": DISPLAY_LABELS[target_class],
                            "target_layer": TARGET_LAYERS[model_type],
                            "precision_policy": precision_policy,
                            "artifact_label": ARTIFACT_LABELS[artifact_label],
                            "panel_png": str(panel_path),
                            **paths,
                        }
                    )
                del model
                tf.keras.backend.clear_session()

    metadata = pd.DataFrame(rows)
    expected_rows = expected_folds * len(DISPLAY_LABELS) * 2 * 2 * samples_per_class
    if len(metadata) != expected_rows:
        raise ValueError(f"Expected {expected_rows} appendix rows; generated {len(metadata)}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = save_paired_comparison_panels(metadata, output_dir)
    metadata.to_csv(output_dir / "gradcam_appendix_metadata.csv", index=False)
    appendix_config = {
        "results_root": str(results_root.resolve()),
        "project_root": str(project_root.resolve()),
        "samples_per_class": samples_per_class,
        "selection_seed": seed,
        "image_size": image_size,
        "overlay_alpha": alpha,
        "expected_folds": expected_folds,
        "selection_rule": "stable SHA-256 within fold/scenario/true class",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metadata_sha256": sha256_file(output_dir / "gradcam_appendix_metadata.csv"),
    }
    (output_dir / "gradcam_appendix_config.json").write_text(
        json.dumps(appendix_config, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced deterministic Grad-CAM appendix panels."
    )
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--samples_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument(
        "--project_root",
        type=Path,
        default=None,
        help="Repository root used to relocate archived absolute data paths.",
    )
    args = parser.parse_args()
    if args.samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    output_dir = args.output_dir or args.results_root / "gradcam_appendix"
    project_root = args.project_root or args.results_root.resolve().parents[1]
    tf.keras.utils.set_random_seed(args.seed)
    metadata = generate_appendix(
        args.results_root,
        output_dir,
        args.samples_per_class,
        args.seed,
        args.image_size,
        args.alpha,
        args.expected_folds,
        project_root,
    )
    print(f"Generated {len(metadata)} Grad-CAM model/sample panels under {output_dir}")


if __name__ == "__main__":
    main()
