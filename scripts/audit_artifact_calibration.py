#!/usr/bin/env python
"""Create artifact calibration previews and distortion statistics.

This utility is deliberately separate from the training pipeline. It reads only
an explicitly supplied development/training partition, does not modify source
images, and does not select a final V2 artifact preset automatically. Candidate
metal-streak previews must be reviewed before the training implementation is
changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from .train_one_case_5fold import (
        IMAGE_EXTENSIONS,
        add_gaussian_noise,
        add_metal_streak,
        add_motion_blur,
        ensure_uint8,
    )
except ImportError:  # Direct execution: python scripts/audit_artifact_calibration.py
    from train_one_case_5fold import (
        IMAGE_EXTENSIONS,
        add_gaussian_noise,
        add_metal_streak,
        add_motion_blur,
        ensure_uint8,
    )

Transform = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def deterministic_rng(seed: int, path: Path, transform_name: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{seed}:{path.as_posix()}:{transform_name}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def scan_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Calibration source does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under {root}")
    return paths


def balanced_sample(paths: Iterable[Path], sample_count: int, seed: int) -> List[Path]:
    """Sample approximately equally from immediate parent class folders."""
    paths = list(paths)
    if sample_count >= len(paths):
        return paths

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent.name].append(path)

    rng = np.random.default_rng(seed)
    selected: List[Path] = []
    per_group = max(1, math.ceil(sample_count / len(grouped)))
    for class_name in sorted(grouped):
        candidates = grouped[class_name]
        indices = rng.choice(len(candidates), size=min(per_group, len(candidates)), replace=False)
        selected.extend(candidates[int(index)] for index in indices)

    if len(selected) > sample_count:
        indices = rng.choice(len(selected), size=sample_count, replace=False)
        selected = [selected[int(index)] for index in indices]
    elif len(selected) < sample_count:
        remaining = sorted(set(paths) - set(selected))
        indices = rng.choice(
            len(remaining), size=min(sample_count - len(selected), len(remaining)), replace=False
        )
        selected.extend(remaining[int(index)] for index in indices)
    return sorted(selected)


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def add_candidate_metal_streak(
    image: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> np.ndarray:
    """Preview-only additive radiopaque streak candidate.

    Unlike the V1 maximum-overlay implementation, the candidate uses an
    additive blurred mask so a line cannot disappear merely because the source
    pixels are already brighter than the sampled overlay. These settings are
    candidates for visual/domain review, not approved training parameters.
    """
    settings = {
        "mild": {"count": 1, "thickness": 2, "delta": 60.0, "sigma": 1.5},
        "moderate": {"count": 2, "thickness": 4, "delta": 100.0, "sigma": 2.5},
        "severe": {"count": 3, "thickness": 6, "delta": 140.0, "sigma": 3.5},
    }
    if severity not in settings:
        raise ValueError(f"Unknown candidate metal severity: {severity}")

    params = settings[severity]
    height, width = image.shape[:2]
    output = image.astype(np.float32)
    for _ in range(int(params["count"])):
        mask = np.zeros((height, width), dtype=np.float32)
        y_start = int(rng.integers(0, height))
        y_end = int(np.clip(y_start + rng.integers(-height // 3, height // 3 + 1), 0, height - 1))
        cv2.line(
            mask,
            (0, y_start),
            (width - 1, y_end),
            1.0,
            int(params["thickness"]),
        )
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(params["sigma"]))
        maximum = float(mask.max())
        if maximum > 0:
            mask /= maximum
        output += mask[..., None] * float(params["delta"])
    return ensure_uint8(output)


def transformations() -> Dict[str, Transform]:
    return {
        "clean": lambda image, rng: image.copy(),
        "motion_k5": lambda image, rng: add_motion_blur(image, kernel_size=5),
        "motion_k9": lambda image, rng: add_motion_blur(image, kernel_size=9),
        "noise_sigma8": lambda image, rng: add_gaussian_noise(image, rng, sigma=8.0),
        "noise_sigma18": lambda image, rng: add_gaussian_noise(image, rng, sigma=18.0),
        "metal_v1_current": lambda image, rng: add_metal_streak(
            image, rng, num_streaks=int(rng.choice([1, 2, 3]))
        ),
        "metal_candidate_mild": lambda image, rng: add_candidate_metal_streak(
            image, rng, "mild"
        ),
        "metal_candidate_moderate": lambda image, rng: add_candidate_metal_streak(
            image, rng, "moderate"
        ),
        "metal_candidate_severe": lambda image, rng: add_candidate_metal_streak(
            image, rng, "severe"
        ),
    }


def distortion_metrics(clean: np.ndarray, transformed: np.ndarray) -> Dict[str, float]:
    difference = transformed.astype(np.float32) - clean.astype(np.float32)
    absolute = np.abs(difference)
    mse = float(np.mean(np.square(difference)))
    return {
        "mae": float(absolute.mean()),
        "rmse": math.sqrt(mse),
        "psnr_db": float(20.0 * math.log10(255.0 / math.sqrt(mse))) if mse > 0 else math.inf,
        "fraction_abs_change_gt_2": float(np.mean(absolute > 2.0)),
        "fraction_abs_change_gt_10": float(np.mean(absolute > 10.0)),
        "maximum_abs_change": float(absolute.max()),
    }


def labeled_tile(image: np.ndarray, label: str, label_height: int = 28) -> np.ndarray:
    tile = np.pad(image, ((label_height, 0), (0, 0), (0, 0)), constant_values=20)
    cv2.putText(
        tile,
        label,
        (5, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def safe_sample_name(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}"


def write_previews(
    paths: List[Path],
    output_dir: Path,
    image_size: int,
    seed: int,
    transforms: Dict[str, Transform],
) -> List[Dict[str, str]]:
    samples_dir = output_dir / "preview_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    contact_rows: List[np.ndarray] = []
    manifest: List[Dict[str, str]] = []

    for path in paths:
        clean = load_image(path, image_size)
        tiles: List[np.ndarray] = []
        row: Dict[str, str] = {"source_path": path.as_posix()}
        for name, transform in transforms.items():
            rng = deterministic_rng(seed, path, name)
            transformed = transform(clean, rng)
            destination = samples_dir / f"{safe_sample_name(path)}__{name}.png"
            cv2.imwrite(str(destination), cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
            row[name] = destination.as_posix()
            tiles.append(labeled_tile(transformed, name))
        contact_rows.append(np.concatenate(tiles, axis=1))
        manifest.append(row)

    contact_sheet = np.concatenate(contact_rows, axis=0)
    cv2.imwrite(
        str(output_dir / "artifact_calibration_contact_sheet.png"),
        cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR),
    )
    return manifest


def write_metrics(
    paths: List[Path],
    output_dir: Path,
    image_size: int,
    seed: int,
    transforms: Dict[str, Transform],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in paths:
        clean = load_image(path, image_size)
        for name, transform in transforms.items():
            rng = deterministic_rng(seed, path, name)
            transformed = transform(clean, rng)
            rows.append(
                {
                    "source_path": path.as_posix(),
                    "class_name": path.parent.name,
                    "transformation": name,
                    **distortion_metrics(clean, transformed),
                }
            )

    raw = pd.DataFrame(rows)
    raw["pixel_identical"] = raw["mae"] == 0.0
    raw["psnr_db_finite"] = raw["psnr_db"].replace([np.inf, -np.inf], np.nan)
    raw.to_csv(output_dir / "artifact_distortion_per_image.csv", index=False)
    summary = (
        raw.groupby("transformation", sort=False)
        .agg(
            n=("source_path", "count"),
            pixel_identical_count=("pixel_identical", "sum"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            psnr_db_finite_mean=("psnr_db_finite", "mean"),
            changed_gt_2_mean=("fraction_abs_change_gt_2", "mean"),
            changed_gt_10_mean=("fraction_abs_change_gt_10", "mean"),
            maximum_abs_change_mean=("maximum_abs_change", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "artifact_distortion_summary.csv", index=False)
    return summary


def write_review_template(output_dir: Path) -> None:
    content = """# Artifact Calibration Review (V2)

This output was generated from a training partition only. No source image was modified, and none of the preview-only metal candidates has been integrated into model training.

## Review instructions

1. Open `artifact_calibration_contact_sheet.png` at full resolution.
2. Compare the candidate artifacts with the clean image in every row.
3. Have the adviser/domain reviewer assess whether the corruption remains plausible for panoramic radiography and whether the TMJ anatomy remains interpretable.
4. Record one decision below. Do not select a preset based on test accuracy.

## Reviewer decision

- Reviewer name/role:
- Review date:
- Motion setting selected:
- Gaussian-noise setting selected:
- Metal setting selected:
- Rejected settings and reason:
- Is the diagnostic anatomy still interpretable? Yes / No
- Approved for V2 pilot training? Yes / No

## Important

High artifact-classification accuracy alone is not evidence of clinical realism. The final selection must balance visibility, plausibility, and preservation of diagnostically relevant anatomy.
"""
    (output_dir / "CALIBRATION_REVIEW.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training-only artifact calibration previews and statistics."
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("data_5_fold/fold_1/train"),
        help="Development/training image root. Do not use an outer test partition.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chapter4_results/artifact_calibration_v2"),
    )
    parser.add_argument("--preview_samples", type=int, default=8)
    parser.add_argument("--metric_samples", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_samples < 1 or args.metric_samples < 1:
        raise ValueError("preview_samples and metric_samples must be positive")
    if "test" in {part.lower() for part in args.source_root.parts}:
        raise ValueError("Calibration must not use a test partition.")

    all_paths = scan_images(args.source_root)
    preview_paths = balanced_sample(all_paths, args.preview_samples, args.seed)
    metric_paths = balanced_sample(all_paths, args.metric_samples, args.seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transforms = transformations()

    preview_manifest = write_previews(
        preview_paths, args.output_dir, args.image_size, args.seed, transforms
    )
    summary = write_metrics(
        metric_paths, args.output_dir, args.image_size, args.seed, transforms
    )
    config = {
        "status": "preview_only_not_approved_for_training",
        "source_root": args.source_root.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "available_source_images": len(all_paths),
        "preview_samples": len(preview_paths),
        "metric_samples": len(metric_paths),
        "image_size": args.image_size,
        "seed": args.seed,
        "transformations": list(transforms),
        "preview_manifest": preview_manifest,
    }
    (args.output_dir / "artifact_calibration_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_review_template(args.output_dir)

    print("\nArtifact distortion summary")
    print(summary.to_string(index=False))
    print(f"\nContact sheet: {(args.output_dir / 'artifact_calibration_contact_sheet.png').resolve()}")
    print(f"Review template: {(args.output_dir / 'CALIBRATION_REVIEW.md').resolve()}")
    print("No training code or source image was modified.")


if __name__ == "__main__":
    main()
