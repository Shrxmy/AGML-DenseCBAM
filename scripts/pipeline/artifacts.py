"""Deterministic image preprocessing and controlled synthetic artifacts."""
from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def add_motion_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return ensure_uint8(cv2.filter2D(image, -1, kernel))


def add_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 12.0,
) -> np.ndarray:
    return ensure_uint8(image.astype(np.float32) + rng.normal(0, sigma, image.shape))


def add_metal_streak_v1(
    image: np.ndarray,
    rng: np.random.Generator,
    num_streaks: int = 2,
) -> np.ndarray:
    # V1 transform retained for calibration comparisons.
    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)
    for _ in range(max(1, num_streaks)):
        overlay = np.zeros_like(output)
        intensity = int(rng.integers(150, 235))
        thickness = int(rng.integers(1, 5))
        cv2.line(
            overlay,
            (int(rng.integers(0, w)), int(rng.integers(0, h))),
            (int(rng.integers(0, w)), int(rng.integers(0, h))),
            (intensity, intensity, intensity),
            thickness,
        )
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=float(rng.uniform(2.0, 5.0)))
        output = np.maximum(output, overlay)
    return ensure_uint8(output)


def add_metal_streak(
    image: np.ndarray,
    rng: np.random.Generator,
    num_streaks: int = 2,
) -> np.ndarray:
    # Add visible localized V2 metal streaks.
    h, w = image.shape[:2]
    output = image.astype(np.float32).copy()
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(max(1, num_streaks)):
        mask = np.zeros((h, w), dtype=np.float32)
        y_start = int(rng.integers(0, h))
        y_end = int(np.clip(y_start + rng.integers(-h // 3, h // 3 + 1), 0, h - 1))
        cv2.line(
            mask,
            (0, y_start),
            (w - 1, y_end),
            1.0,
            int(rng.integers(2, 5)),
        )
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(rng.uniform(1.5, 2.5)))
        maximum = float(mask.max())
        if maximum > 0:
            mask /= maximum
        combined_mask = np.maximum(combined_mask, mask * float(rng.uniform(80.0, 110.0)))

    transformed = ensure_uint8(output + combined_mask[..., None])
    if np.array_equal(transformed, image):
        transformed = ensure_uint8(output - combined_mask[..., None])
    if np.array_equal(transformed, image):
        raise RuntimeError("V2 metal-streak transform produced no pixel change.")
    return transformed


def apply_artifact(
    image: np.ndarray,
    artifact_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if artifact_label == 0:
        return image
    if artifact_label == 1:
        return add_motion_blur(image, kernel_size=int(rng.choice([5, 7, 9])))
    if artifact_label == 2:
        return add_gaussian_noise(image, rng, sigma=float(rng.uniform(8.0, 12.0)))
    if artifact_label == 3:
        return add_metal_streak(image, rng, num_streaks=int(rng.choice([1, 2])))
    raise ValueError(f"Unknown artifact label: {artifact_label}")


def densenet_preprocess(image: np.ndarray) -> np.ndarray:
    # Map image values from [0, 255] to [-1, 1].
    return image.astype(np.float32) / 127.5 - 1.0


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def densenet_torch_preprocess(image: np.ndarray) -> np.ndarray:
    # Keras DenseNet ImageNet weights expect "torch" mode (RGB mean/std).
    return (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def mild_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """V3 training-only geometry and intensity jitter, applied before artifacts."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-7, 7)), float(rng.uniform(0.9, 1.1)))
    matrix[:, 2] += (float(rng.uniform(-0.05, 0.05)) * w, float(rng.uniform(-0.05, 0.05)) * h)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    contrast, brightness = float(rng.uniform(0.9, 1.1)), float(rng.uniform(-0.1, 0.1)) * 255.0
    mean = image.mean()
    return ensure_uint8((image.astype(np.float32) - mean) * contrast + mean + brightness)
