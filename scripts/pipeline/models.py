"""DenseNet201 benchmark and proposed AGML-DenseCBAM model definitions."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import ARTIFACT_LABELS, TMD_LABELS, RunConfig
from .data import TMJSequence


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
            return inputs + tf.cast(attended, inputs.dtype)

        avg_descriptor = self.shared_dense_2(self.shared_dense_1(self.avg_pool(inputs)))
        max_descriptor = self.shared_dense_2(self.shared_dense_1(self.max_pool(inputs)))
        channel_attention = tf.nn.sigmoid(avg_descriptor + max_descriptor)
        channel_attention = tf.reshape(channel_attention, (-1, 1, 1, inputs.shape[-1]))
        x = inputs * tf.cast(channel_attention, inputs.dtype)
        avg_map = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_map = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_attention = self.spatial_conv(tf.concat([avg_map, max_map], axis=-1))
        return x * tf.cast(spatial_attention, x.dtype)


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
    return Model(backbone.input, output, name="DenseNet201_Benchmark_SelfAttention")


def build_proposed_model(config: RunConfig) -> Model:
    backbone = make_backbone(config)
    conv5 = backbone.get_layer("conv5_block32_concat").output

    # TMD branch.
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

    # Pre-CBAM artifact branch.
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

    # Named outputs keep targets and metrics aligned.
    return Model(
        backbone.input,
        {"tmd_output": tmd_output, "artifact_output": artifact_output},
        name="AGML_DenseCBAM",
    )


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


def _training_callbacks(
    checkpoint_path: Path,
    monitor: str,
) -> List[tf.keras.callbacks.Callback]:
    return [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
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


def fit_model(
    model: Model,
    train_gen: TMJSequence,
    val_gen: TMJSequence,
    config: RunConfig,
    checkpoint_path: Path,
) -> pd.DataFrame:
    monitor = "val_tmd_output_loss" if config.model_type == "proposed" else "val_loss"
    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=1,
        callbacks=_training_callbacks(checkpoint_path, monitor),
    )
    return pd.DataFrame(history.history)
