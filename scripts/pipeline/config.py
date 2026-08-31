"""Experiment constants, configuration, reproducibility, and TensorFlow setup."""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from .provenance import sha256_json

tf.config.optimizer.set_jit(False)

TMD_LABELS = ["normal", "subluxation"]
DISPLAY_LABELS = ["Normal", "Subluxation"]
ARTIFACT_LABELS = ["none", "motion_blur", "gaussian_noise", "metal_streak"]
ARTIFACT_PROTOCOL = "v2_moderate_pre_cbam_aux"
TRAINING_PROTOCOL = "v2_single_stage_horizontal_flip"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RunConfig:
    folds_root: Path
    output_dir: Path
    model_type: str
    scenario: str
    image_size: Tuple[int, int]
    batch_size: int
    epochs: int
    learning_rate: float
    l2_strength: float
    random_state: int
    freeze_backbone: bool
    tmd_loss_weight: float
    artifact_loss_weight: float
    fold_limit: int | None
    single_fold: str | None
    verify_integrity: bool
    mixed_precision: bool
    class_weighting: bool
    run_config_sha256: str | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow(config: RunConfig) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"Could not enable memory growth for {gpu}: {exc}")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        print(f"Full TensorFlow determinism could not be enabled: {exc}")

    policy = "mixed_float16" if config.mixed_precision and gpus else "float32"
    mixed_precision.set_global_policy(policy)
    print(f"TensorFlow precision policy: {policy}")


def effective_run_config_sha256(config: RunConfig) -> str:
    if config.run_config_sha256:
        return config.run_config_sha256
    serializable = asdict(config)
    serializable["folds_root"] = str(config.folds_root.resolve())
    serializable["output_dir"] = str(config.output_dir.resolve())
    serializable["image_size"] = list(config.image_size)
    serializable["run_config_sha256"] = None
    return sha256_json(serializable)

