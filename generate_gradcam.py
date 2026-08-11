#!/usr/bin/env python
"""Generate Grad-CAM and optional ROI localization metrics for one final checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from train_one_case_5fold import (
    ARTIFACT_LABELS,
    DISPLAY_LABELS,
    AttentionBlock,
    TMJSequence,
)

TARGET_LAYERS = {
    "benchmark": "benchmark_fusion_conv",
    "proposed": "cbam_refine_conv",
}


def load_input(
    image_path: Path,
    model_type: str,
    scenario: str,
    image_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    frame = pd.DataFrame(
        [{"filepath": str(image_path), "tmd_label": 0, "class_name": "unknown", "split": "test"}]
    )
    generator = TMJSequence(
        frame,
        image_size=(image_size, image_size),
        batch_size=1,
        multi_task=model_type == "proposed",
        scenario=scenario,
        training=False,
        seed=seed,
    )
    batch = generator[0]
    x = batch[0]
    display_rgb = np.clip((x[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    artifact_label = generator.evaluation_artifact_label(str(image_path))
    return x, display_rgb, artifact_label


def gradcam(
    model: tf.keras.Model,
    inputs: np.ndarray,
    target_layer_name: str,
    class_index: int | None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    target_layer = model.get_layer(target_layer_name)
    tmd_output = model.get_layer("tmd_output").output
    gradient_model = tf.keras.Model(model.inputs, [target_layer.output, tmd_output])

    with tf.GradientTape() as tape:
        feature_maps, probabilities = gradient_model(inputs, training=False)
        if class_index is None:
            class_index = int(tf.argmax(probabilities[0]))
        target_score = probabilities[:, class_index]

    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise RuntimeError(f"No gradient reached target layer {target_layer_name!r}.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return np.asarray(heatmap), np.asarray(probabilities[0]), class_index


def overlay_heatmap(display_rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    resized = cv2.resize(heatmap, (display_rgb.shape[1], display_rgb.shape[0]))
    colored_bgr = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(display_rgb, 1.0 - alpha, colored_rgb, alpha, 0)


def load_roi(roi_csv: Path, image_path: Path) -> Tuple[float, float, float, float]:
    matches = []
    with roi_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "x_min", "y_min", "x_max", "y_max"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{roi_csv} must contain columns: {sorted(required)}")
        for row in reader:
            sample_id = row["sample_id"].replace("\\", "/")
            if sample_id in {image_path.name, str(image_path).replace("\\", "/")}:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one ROI row for {image_path}; found {len(matches)}")
    row = matches[0]
    return tuple(float(row[key]) for key in ["x_min", "y_min", "x_max", "y_max"])


def roi_metrics(
    heatmap: np.ndarray,
    roi: Tuple[float, float, float, float],
    original_size: Tuple[int, int],
    analysis_size: Tuple[int, int],
    threshold: float,
) -> Dict[str, float]:
    original_width, original_height = original_size
    width, height = analysis_size
    resized_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    x_min, y_min, x_max, y_max = roi
    coordinates = [
        int(np.clip(round(x_min / original_width * width), 0, width)),
        int(np.clip(round(y_min / original_height * height), 0, height)),
        int(np.clip(round(x_max / original_width * width), 0, width)),
        int(np.clip(round(y_max / original_height * height), 0, height)),
    ]
    sx_min, sy_min, sx_max, sy_max = coordinates
    if sx_max <= sx_min or sy_max <= sy_min:
        raise ValueError(f"ROI has no area after scaling: {roi}")

    roi_mask = np.zeros_like(resized_heatmap, dtype=bool)
    roi_mask[sy_min:sy_max, sx_min:sx_max] = True
    localization_energy = float(
        resized_heatmap[roi_mask].sum() / max(resized_heatmap.sum(), 1e-12)
    )

    attention_mask = resized_heatmap >= threshold
    intersection = np.logical_and(attention_mask, roi_mask).sum()
    union = np.logical_or(attention_mask, roi_mask).sum()
    iou = float(intersection / union) if union else 0.0
    return {
        "localization_energy": localization_energy,
        "gradcam_roi_iou": iou,
        "heatmap_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM from a leakage-safe final checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model_type", choices=["benchmark", "proposed"], required=True)
    parser.add_argument("--scenario", choices=["clean", "artifact_mix"], default="clean")
    parser.add_argument("--class_index", type=int, choices=[0, 1], default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--roi_csv", type=Path, default=None)
    parser.add_argument("--heatmap_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=Path, default=Path("chapter4_results/gradcam"))
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 <= args.heatmap_threshold <= 1:
        raise ValueError("heatmap_threshold must be between 0 and 1")

    tf.keras.utils.set_random_seed(args.seed)
    model = tf.keras.models.load_model(
        args.checkpoint,
        custom_objects={"AttentionBlock": AttentionBlock},
        compile=False,
    )
    inputs, display_rgb, artifact_label = load_input(
        args.image,
        args.model_type,
        args.scenario,
        args.image_size,
        args.seed,
    )
    heatmap, probabilities, class_index = gradcam(
        model,
        inputs,
        TARGET_LAYERS[args.model_type],
        args.class_index,
    )
    raw_outputs = model(inputs, training=False)
    overlay = overlay_heatmap(display_rgb, heatmap, args.alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_type}_{args.scenario}_{args.image.stem}"
    input_path = args.output_dir / f"{stem}_input.png"
    heatmap_path = args.output_dir / f"{stem}_heatmap.png"
    overlay_path = args.output_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(input_path), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), np.uint8(cv2.resize(heatmap, (args.image_size, args.image_size)) * 255))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    original = cv2.imread(str(args.image))
    if original is None:
        raise ValueError(f"Could not read original image: {args.image}")
    result: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "model_type": args.model_type,
        "scenario": args.scenario,
        "target_layer": TARGET_LAYERS[args.model_type],
        "predicted_class_index": class_index,
        "predicted_class": DISPLAY_LABELS[class_index],
        "normal_probability": float(probabilities[0]),
        "subluxation_probability": float(probabilities[1]),
        "artifact_label": ARTIFACT_LABELS[artifact_label],
        "input_png": str(input_path),
        "heatmap_png": str(heatmap_path),
        "overlay_png": str(overlay_path),
    }
    if args.model_type == "proposed":
        artifact_probabilities = np.asarray(raw_outputs["artifact_output"])[0]
        result["predicted_artifact"] = ARTIFACT_LABELS[int(np.argmax(artifact_probabilities))]
        result["artifact_probabilities"] = {
            label: float(probability)
            for label, probability in zip(ARTIFACT_LABELS, artifact_probabilities)
        }
    if args.roi_csv is not None:
        roi = load_roi(args.roi_csv, args.image)
        result["roi_original_pixels"] = roi
        result.update(
            roi_metrics(
                heatmap,
                roi,
                original_size=(original.shape[1], original.shape[0]),
                analysis_size=(args.image_size, args.image_size),
                threshold=args.heatmap_threshold,
            )
        )

    result_path = args.output_dir / f"{stem}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
