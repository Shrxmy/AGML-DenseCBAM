#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import Model, layers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence as KerasSequence, to_categorical

tf.config.optimizer.set_jit(False)

TMD_LABELS = ["normal", "subluxation"]
DISPLAY_LABELS = ["Normal", "Subluxation"]
ARTIFACT_LABELS = ["none", "motion_blur", "gaussian_noise", "metal_streak"]
ARTIFACT_PROTOCOL = "v2_moderate_pre_cbam_aux"
TRAINING_PROTOCOL_V2 = "v2_single_stage_horizontal_flip"
TRAINING_PROTOCOL_V3 = "v3_two_stage_conservative_augmentation"
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
    conservative_augmentation: bool = False
    two_stage_fine_tuning: bool = False
    warmup_epochs: int = 5
    fine_tune_learning_rate: float = 1e-5
    fine_tune_from_layer: str = "conv5_block1_0_bn"
    freeze_batch_norm: bool = True
    evaluation_split: str = "test"
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Dict[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_run_config_sha256(config: RunConfig) -> str:
    if config.run_config_sha256:
        return config.run_config_sha256
    serializable = asdict(config)
    serializable["folds_root"] = str(config.folds_root.resolve())
    serializable["output_dir"] = str(config.output_dir.resolve())
    serializable["image_size"] = list(config.image_size)
    serializable["run_config_sha256"] = None
    return sha256_json(serializable)


def validate_split_integrity(df: pd.DataFrame) -> None:
    """Reject exact-image leakage and conflicting labels before training."""
    audit = df.copy()
    audit["content_sha256"] = audit["filepath"].map(lambda value: sha256_file(Path(value)))

    conflicts = audit.groupby("content_sha256")["tmd_label"].nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index)
    split_counts = audit.groupby("content_sha256")["split"].nunique()
    leaking_hashes = set(split_counts[split_counts > 1].index)

    if conflicting_hashes or leaking_hashes:
        raise ValueError(
            "Dataset integrity check failed: "
            f"{len(leaking_hashes)} exact-image groups cross train/validation/test and "
            f"{len(conflicting_hashes)} groups carry conflicting labels. "
            "Regenerate folds with scripts/make_5fold_dataset.py after reviewing duplicate_audit.csv."
        )


def validate_manifest_integrity(root: Path, expected_image_count: int) -> None:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Integrity verification requires the generated fold manifest: {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"fold_split", "content_sha256", "group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Legacy/incomplete manifest {manifest_path} is missing {sorted(missing)}. "
            "Regenerate folds with the leakage-safe splitter."
        )
    if len(manifest) != expected_image_count:
        raise ValueError(
            f"Manifest/image count mismatch in {root}: {len(manifest)} rows vs "
            f"{expected_image_count} indexed images."
        )
    leaking_groups = manifest.groupby("group_id")["fold_split"].nunique()
    leaking_groups = leaking_groups[leaking_groups > 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"Patient/study group leakage detected in {root}: "
            f"{len(leaking_groups)} groups cross train/validation/test."
        )


def index_split_dataset(root: Path, verify_integrity: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    class_to_idx = {name: idx for idx, name in enumerate(TMD_LABELS)}
    for split in ["train", "validation", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        for class_name in TMD_LABELS:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "filepath": str(path),
                            "split": split,
                            "class_name": class_name,
                            "tmd_label": class_to_idx[class_name],
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {root}")
    if verify_integrity:
        validate_split_integrity(df)
        validate_manifest_integrity(root, expected_image_count=len(df))
    return df


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
    """Preserved V1 maximum-overlay transform for calibration comparisons only."""
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
    """Add localized, visible V2 radiopaque streaks without silently becoming clean."""
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


def apply_conservative_augmentation(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply deterministic, training-only V3 radiographic augmentation.

    One affine resampling combines rotation (±5 degrees), translation (±3%),
    and zoom (0.95-1.05). Contrast is adjusted by ±10% around the image mean.
    Reflection padding avoids artificial black borders. Vertical flips are never
    used because they are anatomically implausible for this task.
    """
    height, width = image.shape[:2]
    angle = float(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.95, 1.05))
    translate_x = float(rng.uniform(-0.03, 0.03) * width)
    translate_y = float(rng.uniform(-0.03, 0.03) * height)
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += translate_x
    matrix[1, 2] += translate_y
    augmented = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    contrast = float(rng.uniform(0.9, 1.1))
    mean = augmented.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return ensure_uint8((augmented.astype(np.float32) - mean) * contrast + mean)


def densenet_preprocess(image: np.ndarray) -> np.ndarray:
    """DenseNet ImageNet preprocessing (torch mode): [0, 255] -> [-1, 1]."""
    return image.astype(np.float32) / 127.5 - 1.0


class TMJSequence(KerasSequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int],
        batch_size: int,
        multi_task: bool,
        scenario: str,
        training: bool,
        seed: int,
        tmd_class_weights: Dict[int, float] | None = None,
        conservative_augmentation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.scenario = scenario
        self.training = training
        self.seed = seed
        self.tmd_class_weights = tmd_class_weights
        self.conservative_augmentation = conservative_augmentation
        self.shuffle_rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))
        self.epoch = -1
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.training:
            self.epoch += 1
            self.shuffle_rng.shuffle(self.indices)

    def _training_rng(self, filepath: str) -> np.random.Generator:
        sample_id = Path(filepath).name
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{sample_id}".encode("utf-8")
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _augmentation_rng(self, filepath: str) -> np.random.Generator:
        """Independent V3 stream so V2 artifact assignments remain unchanged."""
        sample_id = Path(filepath).name
        digest = hashlib.sha256(
            f"v3-augmentation:{self.seed}:{self.epoch}:{sample_id}".encode("utf-8")
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _evaluation_rng(self, filepath: str) -> np.random.Generator:
        # Fold filenames contain the source content hash and remain stable across cases.
        sample_id = Path(filepath).name
        digest = hashlib.sha256(f"{self.seed}:{sample_id}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _artifact_for(self, rng: np.random.Generator) -> int:
        if self.scenario == "clean":
            return 0
        return int(rng.integers(0, len(ARTIFACT_LABELS)))

    def evaluation_artifact_label(self, filepath: str) -> int:
        if self.training:
            raise RuntimeError("Deterministic artifact labels are only available for evaluation generators.")
        return self._artifact_for(self._evaluation_rng(filepath))

    def _load_image(self, filepath: str) -> np.ndarray:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def __getitem__(self, index: int):
        batch_df = self.df.iloc[self.indices[index * self.batch_size : (index + 1) * self.batch_size]]
        images: List[np.ndarray] = []
        tmd_labels: List[int] = []
        artifact_labels: List[int] = []
        for row in batch_df.itertuples(index=False):
            image = self._load_image(row.filepath)
            item_rng = self._training_rng(row.filepath) if self.training else self._evaluation_rng(row.filepath)
            if self.training and item_rng.random() < 0.5:
                image = cv2.flip(image, 1)
            if self.training and self.conservative_augmentation:
                image = apply_conservative_augmentation(
                    image,
                    self._augmentation_rng(row.filepath),
                )
            artifact_label = self._artifact_for(item_rng)
            image = densenet_preprocess(apply_artifact(image, artifact_label, item_rng))
            images.append(image)
            tmd_labels.append(int(row.tmd_label))
            artifact_labels.append(int(artifact_label))

        x = np.stack(images)
        y_tmd = to_categorical(np.array(tmd_labels), num_classes=len(TMD_LABELS))
        y_artifact = to_categorical(np.array(artifact_labels), num_classes=len(ARTIFACT_LABELS))
        if self.multi_task:
            targets = {"tmd_output": y_tmd, "artifact_output": y_artifact}
            if self.training and self.tmd_class_weights:
                sample_weights = {
                    "tmd_output": np.asarray(
                        [self.tmd_class_weights[label] for label in tmd_labels],
                        dtype=np.float32,
                    ),
                    "artifact_output": np.ones(len(tmd_labels), dtype=np.float32),
                }
                return x, targets, sample_weights
            return x, targets

        if self.training and self.tmd_class_weights:
            sample_weights = np.asarray(
                [self.tmd_class_weights[label] for label in tmd_labels],
                dtype=np.float32,
            )
            return x, y_tmd, sample_weights
        return x, y_tmd


@tf.keras.utils.register_keras_serializable(package="AGML")
class AttentionBlock(layers.Layer):
    def __init__(self, attention_type: str = "cbam", reduction_ratio: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention_type = attention_type.lower()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        filters = int(input_shape[-1])
        reduced_filters = max(filters // self.reduction_ratio, 1)
        if self.attention_type == "self":
            qk_filters = max(filters // 8, 1)
            self.query_conv = layers.Conv2D(qk_filters, 1, padding="same")
            self.key_conv = layers.Conv2D(qk_filters, 1, padding="same")
            self.value_conv = layers.Conv2D(filters, 1, padding="same")
        elif self.attention_type == "cbam":
            self.avg_pool = layers.GlobalAveragePooling2D()
            self.max_pool = layers.GlobalMaxPooling2D()
            self.shared_dense_1 = layers.Dense(reduced_filters, activation="relu")
            self.shared_dense_2 = layers.Dense(filters)
            self.spatial_conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid")
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "attention_type": self.attention_type,
                "reduction_ratio": self.reduction_ratio,
            }
        )
        return config

    def call(self, inputs):
        if self.attention_type == "self":
            shape = tf.shape(inputs)
            batch_size, height, width = shape[0], shape[1], shape[2]
            channels = inputs.shape[-1]
            q = tf.reshape(self.query_conv(inputs), [batch_size, height * width, -1])
            k = tf.reshape(self.key_conv(inputs), [batch_size, height * width, -1])
            v = tf.reshape(self.value_conv(inputs), [batch_size, height * width, channels])
            attention_scores = tf.matmul(q, k, transpose_b=True)
            scale = tf.math.sqrt(tf.cast(tf.shape(k)[-1], attention_scores.dtype))
            attention_scores = attention_scores / scale
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended = tf.matmul(attention_weights, v)
            attended = tf.reshape(attended, [batch_size, height, width, channels])
            return inputs + attended

        avg_descriptor = self.shared_dense_2(self.shared_dense_1(self.avg_pool(inputs)))
        max_descriptor = self.shared_dense_2(self.shared_dense_1(self.max_pool(inputs)))
        channel_attention = tf.nn.sigmoid(avg_descriptor + max_descriptor)
        channel_attention = tf.reshape(channel_attention, (-1, 1, 1, inputs.shape[-1]))
        x = inputs * channel_attention
        avg_map = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_map = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_attention = self.spatial_conv(tf.concat([avg_map, max_map], axis=-1))
        return x * spatial_attention


def make_backbone(config: RunConfig) -> Model:
    backbone = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.image_size, 3),
        pooling=None,
    )
    backbone.trainable = not (config.freeze_backbone or config.two_stage_fine_tuning)
    return backbone


def build_benchmark_model(config: RunConfig) -> Model:
    backbone = make_backbone(config)
    pool3 = backbone.get_layer("pool3_relu").output
    pool3_att = AttentionBlock("self", name="benchmark_self_attention")(pool3)
    pool3_down = layers.AveragePooling2D(pool_size=4, strides=4, name="benchmark_pool3_downsample")(pool3_att)
    conv5 = backbone.get_layer("conv5_block32_concat").output
    pool3_proj = layers.Conv2D(int(conv5.shape[-1]), 1, padding="same", name="benchmark_pool3_projection")(pool3_down)
    fused = layers.Concatenate(name="benchmark_fused_features")([conv5, pool3_proj])
    fused = layers.Conv2D(1024, 1, activation="relu", padding="same", name="benchmark_fusion_conv")(fused)
    x = layers.GlobalAveragePooling2D(name="benchmark_gap")(fused)
    x = layers.Dense(1024, activation="relu", kernel_regularizer=l2(config.l2_strength), name="benchmark_fc1")(x)
    x = layers.Dropout(0.5, name="benchmark_dropout")(x)
    x = layers.BatchNormalization(name="benchmark_bn")(x)
    x = layers.Dense(128, activation="relu", name="benchmark_fc2")(x)
    output = layers.Dense(len(TMD_LABELS), activation="softmax", dtype="float32", name="tmd_output")(x)
    model = Model(backbone.input, output, name="DenseNet201_Benchmark_SelfAttention")
    model._agml_backbone_layer_names = tuple(layer.name for layer in backbone.layers)  # type: ignore[attr-defined]
    return model


def build_proposed_model(config: RunConfig) -> Model:
    backbone = make_backbone(config)
    conv5 = backbone.get_layer("conv5_block32_concat").output

    # TMD classification uses the CBAM-refined representation.
    attended = AttentionBlock("cbam", name="cbam_attention")(conv5)
    tmd_features = layers.GlobalAveragePooling2D(name="tmd_gap")(attended)

    primary = layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=l2(config.l2_strength),
        name="tmd_fc1",
    )(tmd_features)
    primary = layers.Dropout(0.5, name="tmd_dropout")(primary)
    primary = layers.BatchNormalization(name="tmd_bn")(primary)
    primary = layers.Dense(128, activation="relu", name="tmd_fc2")(primary)
    tmd_output = layers.Dense(
        len(TMD_LABELS),
        activation="softmax",
        dtype="float32",
        name="tmd_output",
    )(primary)

    # Artifact detection branches before CBAM so attention suppression cannot erase
    # localized corruption evidence. Max pooling complements average pooling for
    # sparse streaks while retaining a shared DenseNet201 encoder.
    artifact_average = layers.GlobalAveragePooling2D(name="artifact_gap")(conv5)
    artifact_maximum = layers.GlobalMaxPooling2D(name="artifact_gmp")(conv5)
    artifact_features = layers.Concatenate(name="artifact_pooled_features")(
        [artifact_average, artifact_maximum]
    )
    auxiliary = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(config.l2_strength),
        name="artifact_fc1",
    )(artifact_features)
    auxiliary = layers.Dropout(0.3, name="artifact_dropout")(auxiliary)
    artifact_output = layers.Dense(
        len(ARTIFACT_LABELS),
        activation="softmax",
        dtype="float32",
        name="artifact_output",
    )(auxiliary)

    # Explicit output mapping prevents metric/target ambiguity across Keras versions.
    model = Model(
        backbone.input,
        {"tmd_output": tmd_output, "artifact_output": artifact_output},
        name="AGML_DenseCBAM",
    )
    model._agml_backbone_layer_names = tuple(layer.name for layer in backbone.layers)  # type: ignore[attr-defined]
    return model


def compile_model(model: Model, config: RunConfig) -> None:
    if config.model_type == "proposed":
        model.compile(
            optimizer=Adam(config.learning_rate),
            loss={"tmd_output": "categorical_crossentropy", "artifact_output": "categorical_crossentropy"},
            loss_weights={"tmd_output": config.tmd_loss_weight, "artifact_output": config.artifact_loss_weight},
            metrics={
                "tmd_output": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
                "artifact_output": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
            },
            jit_compile=False,
        )
        return

    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )


def _connected_backbone_layers(model: Model) -> List[layers.Layer]:
    """Return only DenseNet layers connected to this task model, in backbone order."""
    backbone_names = getattr(model, "_agml_backbone_layer_names", None)
    if not backbone_names:
        raise ValueError("Model is missing explicit DenseNet backbone metadata.")
    model_layers = {layer.name: layer for layer in model.layers}
    connected = [model_layers[name] for name in backbone_names if name in model_layers]
    if not connected or connected[-1].name != "conv5_block32_concat":
        raise ValueError("Connected DenseNet backbone does not end at conv5_block32_concat.")
    return connected


def validate_fine_tune_boundary(model: Model, config: RunConfig) -> None:
    """Reject an invalid V3 boundary before spending time on warm-up training."""
    names = [layer.name for layer in _connected_backbone_layers(model)]
    if config.fine_tune_from_layer not in names:
        raise ValueError(
            f"Fine-tune boundary {config.fine_tune_from_layer!r} was not found "
            "in the connected DenseNet201 backbone."
        )


def set_backbone_frozen(model: Model) -> None:
    """Freeze only DenseNet layers while leaving all task-specific heads unchanged."""
    for layer in _connected_backbone_layers(model):
        layer.trainable = False


def configure_selective_fine_tuning(model: Model, config: RunConfig) -> Dict[str, int]:
    """Unfreeze the final DenseNet block without changing task-head trainability."""
    if config.freeze_backbone:
        raise ValueError("--freeze_backbone cannot be combined with two-stage fine-tuning.")
    backbone_layers = _connected_backbone_layers(model)
    names = [layer.name for layer in backbone_layers]
    validate_fine_tune_boundary(model, config)
    start = names.index(config.fine_tune_from_layer)

    counts = {
        "backbone_layers": len(backbone_layers),
        "unfrozen_backbone_layers": 0,
        "frozen_batch_norm_layers": 0,
    }
    for index, layer in enumerate(backbone_layers):
        should_train = index >= start
        if should_train and config.freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            should_train = False
            counts["frozen_batch_norm_layers"] += 1
        layer.trainable = should_train
        if should_train:
            counts["unfrozen_backbone_layers"] += 1
    return counts


def _training_callbacks(
    checkpoint_path: Path,
    monitor: str,
    save_weights_only: bool = False,
    early_stopping: bool = True,
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=save_weights_only,
            verbose=1,
        )
    ]
    if early_stopping:
        callbacks.extend(
            [
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=5,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor=monitor,
                    mode="min",
                    factor=0.1,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1,
                ),
            ]
        )
    return callbacks


def fit_model(
    model: Model,
    train_gen: TMJSequence,
    val_gen: TMJSequence,
    config: RunConfig,
    checkpoint_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fit using either the preserved V2 schedule or locked two-stage V3 schedule."""
    monitor = "val_tmd_output_loss" if config.model_type == "proposed" else "val_loss"
    if not config.two_stage_fine_tuning:
        history = model.fit(
            train_gen,
            epochs=config.epochs,
            validation_data=val_gen,
            verbose=1,
            callbacks=_training_callbacks(checkpoint_path, monitor),
        )
        history_df = pd.DataFrame(history.history)
        return history_df, {
            "selected_stage": "single_stage",
            "stage1_epochs_ran": 0,
            "stage2_epochs_ran": len(history_df),
            "fine_tuning": None,
        }

    if not 1 <= config.warmup_epochs < config.epochs:
        raise ValueError("Two-stage training requires 1 <= warmup_epochs < total epochs.")
    validate_fine_tune_boundary(model, config)

    stage1_path = checkpoint_path.with_name(checkpoint_path.stem + "_stage1.weights.h5")
    stage2_path = checkpoint_path.with_name(checkpoint_path.stem + "_stage2.weights.h5")
    print(f"V3 stage 1: frozen DenseNet201 warm-up for {config.warmup_epochs} epochs.")
    stage1_history = model.fit(
        train_gen,
        epochs=config.warmup_epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=_training_callbacks(
            stage1_path,
            monitor,
            save_weights_only=True,
            early_stopping=False,
        ),
    )
    stage1_df = pd.DataFrame(stage1_history.history)
    if "learning_rate" not in stage1_df:
        stage1_df["learning_rate"] = config.learning_rate
    stage1_df.insert(0, "stage_epoch", np.arange(1, len(stage1_df) + 1))
    stage1_df.insert(0, "training_stage", "frozen_warmup")

    # Stage 2 must begin from the best warm-up checkpoint, not merely the final
    # warm-up epoch, because validation loss can worsen before epoch 5.
    model.load_weights(stage1_path)
    fine_tuning = configure_selective_fine_tuning(model, config)
    print("V3 stage 2 trainability:", fine_tuning)
    stage2_config = RunConfig(
        **{
            **asdict(config),
            "learning_rate": config.fine_tune_learning_rate,
        }
    )
    compile_model(model, stage2_config)
    print(
        f"V3 stage 2: fine-tuning from {config.fine_tune_from_layer} at "
        f"learning rate {config.fine_tune_learning_rate:g}."
    )
    stage2_history = model.fit(
        train_gen,
        initial_epoch=config.warmup_epochs,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=_training_callbacks(
            stage2_path,
            monitor,
            save_weights_only=True,
            early_stopping=True,
        ),
    )
    stage2_df = pd.DataFrame(stage2_history.history)
    if "learning_rate" not in stage2_df:
        stage2_df["learning_rate"] = config.fine_tune_learning_rate
    stage2_df.insert(0, "stage_epoch", np.arange(1, len(stage2_df) + 1))
    stage2_df.insert(0, "training_stage", "selective_fine_tuning")

    stage1_best = float(stage1_df[monitor].min())
    stage2_best = float(stage2_df[monitor].min())
    selected_stage = "frozen_warmup" if stage1_best <= stage2_best else "selective_fine_tuning"
    selected_path = stage1_path if selected_stage == "frozen_warmup" else stage2_path
    if selected_stage == "frozen_warmup":
        set_backbone_frozen(model)
        compile_model(model, config)
    model.load_weights(selected_path)
    model.save(checkpoint_path)
    stage1_path.unlink(missing_ok=True)
    stage2_path.unlink(missing_ok=True)

    history_df = pd.concat([stage1_df, stage2_df], ignore_index=True)
    return history_df, {
        "selected_stage": selected_stage,
        "stage1_epochs_ran": len(stage1_df),
        "stage2_epochs_ran": len(stage2_df),
        "stage1_best_validation_loss": stage1_best,
        "stage2_best_validation_loss": stage2_best,
        "fine_tuning": fine_tuning,
        "checkpoint_purpose": "inference_and_gradcam",
    }


def unpack_batch(batch):
    """Return inputs and targets from a Keras batch with optional sample weights."""
    if len(batch) == 3:
        inputs, targets, _ = batch
        return inputs, targets
    return batch


def balanced_class_weights(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts().to_dict()
    total = len(labels)
    if set(counts) != set(range(len(TMD_LABELS))):
        raise ValueError(f"Training split must contain every TMD class; found counts {counts}")
    return {
        label: total / (len(TMD_LABELS) * count)
        for label, count in counts.items()
    }


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


def run_one_fold(fold_root: Path, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Running {config.model_type} | {config.scenario} | {fold_root.name} ===")
    seed_everything(config.random_state)

    df = index_split_dataset(fold_root, verify_integrity=config.verify_integrity)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    evaluation_df = val_df if config.evaluation_split == "validation" else test_df
    print(df.groupby(["split", "class_name"]).size())
    print(f"Evaluation split: {config.evaluation_split} ({len(evaluation_df)} images)")

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
        conservative_augmentation=config.conservative_augmentation,
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
    history_df, training_metadata = fit_model(
        model,
        train_gen,
        val_gen,
        config,
        checkpoint_path,
    )

    # Warm up graph tracing before measuring end-to-end batch inference.
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
                "epochs_ran": len(history_df),
                "evaluation_split": config.evaluation_split,
                "training_protocol": (
                    TRAINING_PROTOCOL_V3
                    if config.two_stage_fine_tuning and config.conservative_augmentation
                    else TRAINING_PROTOCOL_V2
                ),
                "conservative_augmentation": config.conservative_augmentation,
                "two_stage_fine_tuning": config.two_stage_fine_tuning,
                "selected_stage": training_metadata["selected_stage"],
                "stage1_epochs_ran": training_metadata["stage1_epochs_ran"],
                "stage2_epochs_ran": training_metadata["stage2_epochs_ran"],
                "stage1_best_validation_loss": training_metadata.get(
                    "stage1_best_validation_loss", np.nan
                ),
                "stage2_best_validation_loss": training_metadata.get(
                    "stage2_best_validation_loss", np.nan
                ),
                "unfrozen_backbone_layers": (
                    training_metadata.get("fine_tuning") or {}
                ).get("unfrozen_backbone_layers", 0),
                "frozen_backbone_batch_norm_layers": (
                    training_metadata.get("fine_tuning") or {}
                ).get("frozen_batch_norm_layers", 0),
                "tensorflow_version": tf.__version__,
                "precision_policy": mixed_precision.global_policy().name,
                "artifact_protocol": ARTIFACT_PROTOCOL,
                "class_weighting": config.class_weighting,
                "fold_manifest_sha256": fold_manifest_sha256,
                "training_script_sha256": sha256_file(Path(__file__).resolve()),
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
    if config.evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'")
    if config.freeze_backbone and config.two_stage_fine_tuning:
        raise ValueError("--freeze_backbone cannot be combined with two-stage fine-tuning.")
    if config.conservative_augmentation != config.two_stage_fine_tuning:
        raise ValueError(
            "Locked V3 requires --conservative_augmentation and "
            "--two_stage_fine_tuning together."
        )

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
        serializable["training_protocol"] = (
            TRAINING_PROTOCOL_V3
            if config.two_stage_fine_tuning
            else TRAINING_PROTOCOL_V2
        )
        serializable["training_script_sha256"] = sha256_file(Path(__file__).resolve())
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


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train one thesis case across 5 folds.")
    parser.add_argument("--folds_root", type=Path, default=Path("data_5_fold"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_type", choices=["benchmark", "proposed"], default="proposed")
    parser.add_argument("--scenario", choices=["clean", "artifact_mix"], default="artifact_mix")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--l2_strength",
        "--weight_decay",
        dest="l2_strength",
        type=float,
        default=1e-2,
        help="L2 kernel regularization strength. --weight_decay is retained as a deprecated alias.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--tmd_loss_weight", type=float, default=1.0)
    parser.add_argument("--artifact_loss_weight", type=float, default=0.3)
    parser.add_argument("--fold_limit", type=int, default=None, help="Use 1 for smoke test; omit for all folds.")
    parser.add_argument("--single_fold", type=str, default=None, help="Run only one named fold, e.g. fold_2. Used by isolated runner.")
    parser.add_argument(
        "--skip_integrity_check",
        action="store_true",
        help="Debug only: bypass exact-image leakage validation. Never use for final thesis runs.",
    )
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed_float16 when a GPU is available.",
    )
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply balanced TMD sample weights to the training loss.",
    )
    parser.add_argument(
        "--conservative_augmentation",
        action="store_true",
        help="Enable locked V3 training-only affine and contrast augmentation.",
    )
    parser.add_argument(
        "--two_stage_fine_tuning",
        action="store_true",
        help="Enable frozen warm-up followed by selective final-block fine-tuning.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--fine_tune_learning_rate", type=float, default=1e-5)
    parser.add_argument("--fine_tune_from_layer", type=str, default="conv5_block1_0_bn")
    parser.add_argument(
        "--freeze_batch_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DenseNet BatchNorm statistics frozen during selective fine-tuning.",
    )
    parser.add_argument(
        "--evaluation_split",
        choices=["validation", "test"],
        default="test",
        help="Use validation for development smoke tests; final runs must use test.",
    )
    parser.add_argument(
        "--run_config_sha256",
        type=str,
        default=None,
        help="Canonical isolated-run fingerprint supplied by the runner.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path("chapter4_results") / f"{args.model_type}_{args.scenario}"
    return RunConfig(
        folds_root=args.folds_root,
        output_dir=output_dir,
        model_type=args.model_type,
        scenario=args.scenario,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        random_state=args.seed,
        freeze_backbone=args.freeze_backbone,
        tmd_loss_weight=args.tmd_loss_weight,
        artifact_loss_weight=args.artifact_loss_weight,
        fold_limit=args.fold_limit,
        single_fold=args.single_fold,
        verify_integrity=not args.skip_integrity_check,
        mixed_precision=args.mixed_precision,
        class_weighting=args.class_weighting,
        conservative_augmentation=args.conservative_augmentation,
        two_stage_fine_tuning=args.two_stage_fine_tuning,
        warmup_epochs=args.warmup_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        fine_tune_from_layer=args.fine_tune_from_layer,
        freeze_batch_norm=args.freeze_batch_norm,
        evaluation_split=args.evaluation_split,
        run_config_sha256=args.run_config_sha256,
    )


if __name__ == "__main__":
    run_case(parse_args())
