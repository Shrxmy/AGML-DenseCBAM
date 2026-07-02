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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
tf.config.optimizer.set_jit(False)
try:
    mixed_precision.set_global_policy("mixed_float16")
except Exception as exc:
    print(f"Mixed precision not enabled: {exc}")

TMD_LABELS = ["normal", "subluxation"]
DISPLAY_LABELS = ["Normal", "Subluxation"]
ARTIFACT_LABELS = ["none", "motion_blur", "gaussian_noise", "metal_streak"]
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
    weight_decay: float
    random_state: int
    freeze_backbone: bool
    tmd_loss_weight: float
    artifact_loss_weight: float
    fold_limit: int | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"Could not enable memory growth for {gpu}: {exc}")


def index_split_dataset(root: Path) -> pd.DataFrame:
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
    return df


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def add_motion_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return ensure_uint8(cv2.filter2D(image, -1, kernel))


def add_gaussian_noise(image: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    return ensure_uint8(image.astype(np.float32) + np.random.normal(0, sigma, image.shape))


def add_metal_streak(image: np.ndarray, num_streaks: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    output = image.copy().astype(np.float32)
    for _ in range(max(1, num_streaks)):
        overlay = np.zeros_like(output)
        intensity = np.random.randint(150, 235)
        thickness = np.random.randint(1, 5)
        cv2.line(
            overlay,
            (np.random.randint(0, w), np.random.randint(0, h)),
            (np.random.randint(0, w), np.random.randint(0, h)),
            (intensity, intensity, intensity),
            thickness,
        )
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=float(np.random.uniform(2.0, 5.0)))
        output = np.maximum(output, overlay)
    return ensure_uint8(output)


def apply_artifact(image: np.ndarray, artifact_label: int) -> np.ndarray:
    if artifact_label == 0:
        return image
    if artifact_label == 1:
        return add_motion_blur(image, kernel_size=int(np.random.choice([5, 7, 9, 11])))
    if artifact_label == 2:
        return add_gaussian_noise(image, sigma=float(np.random.uniform(8.0, 18.0)))
    if artifact_label == 3:
        return add_metal_streak(image, num_streaks=int(np.random.choice([1, 2, 3])))
    raise ValueError(f"Unknown artifact label: {artifact_label}")


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
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.scenario = scenario
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.training:
            self.rng.shuffle(self.indices)

    def _artifact_for(self, filepath: str) -> int:
        if self.scenario == "clean":
            return 0
        if self.training:
            return int(self.rng.integers(0, len(ARTIFACT_LABELS)))
        digest = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        return int(digest, 16) % len(ARTIFACT_LABELS)

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
            if self.training and self.rng.random() < 0.5:
                image = cv2.flip(image, 1)
            artifact_label = self._artifact_for(row.filepath)
            image = apply_artifact(image, artifact_label).astype(np.float32) / 255.0
            images.append(image)
            tmd_labels.append(int(row.tmd_label))
            artifact_labels.append(int(artifact_label))

        x = np.stack(images)
        y_tmd = to_categorical(np.array(tmd_labels), num_classes=len(TMD_LABELS))
        y_artifact = to_categorical(np.array(artifact_labels), num_classes=len(ARTIFACT_LABELS))
        if self.multi_task:
            return x, {"tmd_output": y_tmd, "artifact_output": y_artifact}
        return x, y_tmd


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

    def call(self, inputs):
        if self.attention_type == "self":
            shape = tf.shape(inputs)
            batch_size, height, width = shape[0], shape[1], shape[2]
            channels = inputs.shape[-1]
            q = tf.reshape(self.query_conv(inputs), [batch_size, height * width, -1])
            k = tf.reshape(self.key_conv(inputs), [batch_size, height * width, -1])
            v = tf.reshape(self.value_conv(inputs), [batch_size, height * width, channels])
            attention_scores = tf.matmul(q, k, transpose_b=True)
            attention_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
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
    backbone.trainable = not config.freeze_backbone
    return backbone


def build_benchmark_model(config: RunConfig) -> Model:
    backbone = make_backbone(config)
    pool3 = backbone.get_layer("pool3_relu").output
    pool3_att = AttentionBlock("self", name="benchmark_self_attention")(pool3)
    pool3_down = layers.AveragePooling2D(pool_size=2, name="benchmark_pool3_downsample")(pool3_att)
    conv5 = backbone.get_layer("conv5_block32_concat").output
    pool3_proj = layers.Conv2D(int(conv5.shape[-1]), 1, padding="same", name="benchmark_pool3_projection")(pool3_down)
    fused = layers.Concatenate(name="benchmark_fused_features")([conv5, pool3_proj])
    fused = layers.Conv2D(1024, 1, activation="relu", padding="same", name="benchmark_fusion_conv")(fused)
    x = layers.GlobalAveragePooling2D(name="benchmark_gap")(fused)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(config.weight_decay), name="benchmark_fc1")(x)
    x = layers.Dropout(0.5, name="benchmark_dropout")(x)
    x = layers.BatchNormalization(name="benchmark_bn")(x)
    x = layers.Dense(128, activation="relu", name="benchmark_fc2")(x)
    output = layers.Dense(len(TMD_LABELS), activation="softmax", dtype="float32", name="tmd_output")(x)
    return Model(backbone.input, output, name="DenseNet201_Benchmark_SelfAttention")


def build_proposed_model(config: RunConfig) -> Model:
    backbone = make_backbone(config)
    conv5 = backbone.get_layer("conv5_block32_concat").output
    x = AttentionBlock("cbam", name="cbam_attention")(conv5)
    x = layers.Conv2D(1024, 1, activation="relu", padding="same", name="cbam_refine_conv")(x)
    x = layers.GlobalAveragePooling2D(name="shared_gap")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(config.weight_decay), name="shared_fc1")(x)
    x = layers.Dropout(0.5, name="shared_dropout")(x)
    x = layers.BatchNormalization(name="shared_bn")(x)
    x = layers.Dense(128, activation="relu", name="shared_fc2")(x)
    tmd_output = layers.Dense(len(TMD_LABELS), activation="softmax", dtype="float32", name="tmd_output")(x)
    artifact_output = layers.Dense(len(ARTIFACT_LABELS), activation="softmax", dtype="float32", name="artifact_output")(x)
    return Model(backbone.input, [tmd_output, artifact_output], name="AGMTL_DenseCBAM")


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


def collect_predictions(model: Model, generator: TMJSequence, multi_task: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []
    for i in range(len(generator)):
        x, y = generator[i]
        pred = model.predict(x, verbose=0)
        pred_tmd = pred[0] if multi_task else pred
        true_tmd = y["tmd_output"] if multi_task else y
        y_true.extend(np.argmax(true_tmd, axis=1).tolist())
        y_pred.extend(np.argmax(pred_tmd, axis=1).tolist())
        y_conf.extend(np.max(pred_tmd, axis=1).tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def run_one_fold(fold_root: Path, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Running {config.model_type} | {config.scenario} | {fold_root.name} ===")
    seed_everything(config.random_state)

    df = index_split_dataset(fold_root)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(df.groupby(["split", "class_name"]).size())

    multi_task = config.model_type == "proposed"
    train_gen = TMJSequence(train_df, config.image_size, config.batch_size, multi_task, config.scenario, True, config.random_state)
    val_gen = TMJSequence(val_df, config.image_size, config.batch_size, multi_task, config.scenario, False, config.random_state)
    test_gen = TMJSequence(test_df, config.image_size, config.batch_size, multi_task, config.scenario, False, config.random_state)

    model = build_proposed_model(config) if multi_task else build_benchmark_model(config)
    compile_model(model, config)

    case_dir = config.output_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = case_dir / f"{fold_root.name}_{config.model_type}_{config.scenario}_best.keras"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=callbacks,
    )

    start = time.perf_counter()
    y_true, y_pred, y_conf = collect_predictions(model, test_gen, multi_task)
    elapsed = time.perf_counter() - start
    images_per_second = len(y_true) / max(elapsed, 1e-8)
    latency_ms = 1000.0 / max(images_per_second, 1e-8)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
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
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "images_per_second": images_per_second,
                "latency_ms": latency_ms,
                "epochs_ran": len(history.history.get("loss", [])),
            }
        ]
    )
    cm_df = pd.DataFrame(cm, index=DISPLAY_LABELS, columns=DISPLAY_LABELS)
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "confidence": y_conf})
    history_df = pd.DataFrame(history.history)

    result_df.to_csv(case_dir / f"{fold_root.name}_results.csv", index=False)
    cm_df.to_csv(case_dir / f"{fold_root.name}_confusion_matrix.csv")
    pred_df.to_csv(case_dir / f"{fold_root.name}_predictions.csv", index=False)
    history_df.to_csv(case_dir / f"{fold_root.name}_history.csv", index=False)

    print(result_df)
    print(cm_df)

    tf.keras.backend.clear_session()
    gc.collect()
    return result_df, cm_df, pred_df, history_df


def run_case(config: RunConfig) -> pd.DataFrame:
    enable_memory_growth()
    if config.model_type not in {"benchmark", "proposed"}:
        raise ValueError("model_type must be 'benchmark' or 'proposed'")
    if config.scenario not in {"clean", "artifact_mix"}:
        raise ValueError("scenario must be 'clean' or 'artifact_mix'")

    fold_dirs = sorted(
        [p for p in config.folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
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
        json.dump(serializable, f, indent=2)

    for fold_root in fold_dirs:
        result_df, cm_df, _, _ = run_one_fold(fold_root, config)
        result_rows.append(result_df)
        pooled_cm = cm_df.copy() if pooled_cm is None else pooled_cm.add(cm_df, fill_value=0)

    combined = pd.concat(result_rows, ignore_index=True)
    combined.to_csv(config.output_dir / "all_fold_results.csv", index=False)

    summary = combined[
        ["accuracy", "precision", "recall", "specificity", "f1", "images_per_second", "latency_ms", "epochs_ran"]
    ].agg(["mean", "std"]).T.reset_index(names="metric")
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
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--tmd_loss_weight", type=float, default=1.0)
    parser.add_argument("--artifact_loss_weight", type=float, default=0.35)
    parser.add_argument("--fold_limit", type=int, default=None, help="Use 1 for smoke test; omit for all folds.")
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
        weight_decay=args.weight_decay,
        random_state=args.seed,
        freeze_backbone=args.freeze_backbone,
        tmd_loss_weight=args.tmd_loss_weight,
        artifact_loss_weight=args.artifact_loss_weight,
        fold_limit=args.fold_limit,
    )


if __name__ == "__main__":
    run_case(parse_args())
