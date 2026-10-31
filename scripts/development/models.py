"""Proposed-only 2x2 architecture; TensorFlow is imported only inside functions.

The default build has no fusion path, staged fine tuning, or head redesign. The V4
candidates (A: multiscale_gated, B: densenet121, C: artifact_gated) stay validation-only until one
passes the screen. ``weights=None`` is supported for structure checks.
"""
from __future__ import annotations

from .config import DevelopmentConfig


def build_multiscale_gated_model(config: DevelopmentConfig, weights: str | None):
    # Candidate A. Same heads as the default model, with a new path into CBAM.
    import tensorflow as tf
    from tensorflow.keras import Model, layers
    from tensorflow.keras.regularizers import l2
    from scripts.pipeline.config import ARTIFACT_LABELS, TMD_LABELS
    from scripts.pipeline.models import AttentionBlock

    backbone = tf.keras.applications.DenseNet201(
        include_top=False, weights=weights,
        input_shape=(*config.input_hw, 3), pooling=None,
    )
    backbone.trainable = True
    conv5 = backbone.get_layer("conv5_block32_concat").output

    # Pre-CBAM artifact branch, built first because its prediction drives the gate.
    artifact_average = layers.GlobalAveragePooling2D(name="artifact_gap")(conv5)
    artifact_maximum = layers.GlobalMaxPooling2D(name="artifact_gmp")(conv5)
    artifact_features = layers.Concatenate(name="artifact_pooled_features")([artifact_average, artifact_maximum])
    auxiliary = layers.Dense(
        256, activation="relu", kernel_regularizer=l2(config.l2_strength), name="artifact_fc1",
    )(artifact_features)
    auxiliary = layers.Dropout(0.3, name="artifact_dropout")(auxiliary)
    artifact_output = layers.Dense(
        len(ARTIFACT_LABELS), activation="softmax", dtype="float32", name="artifact_output",
    )(auxiliary)

    if config.candidate == "artifact_gated":
        # Candidate C is candidate A without pool3, which kept A from learning.
        fused = conv5
    else:
        # Add finer pool3 detail, using the benchmark's fusion shape.
        pool3 = backbone.get_layer("pool3_relu").output
        pool3 = layers.AveragePooling2D(pool_size=4, strides=4, name="v4_pool3_downsample")(pool3)
        pool3 = layers.Conv2D(int(conv5.shape[-1]), 1, padding="same", name="v4_pool3_projection")(pool3)
        fused = layers.Concatenate(name="v4_fused_features")([conv5, pool3])
    fused = layers.Conv2D(1024, 1, activation="relu", padding="same", name="v4_fusion_conv")(fused)

    # The artifact prediction scales each channel. The added path means a gate
    # near zero leaves the fused features unchanged.
    gate = layers.Dense(1024, activation="sigmoid", name="v4_artifact_gate")(artifact_output)
    gate = layers.Reshape((1, 1, 1024), name="v4_gate_reshape")(gate)
    scaled = layers.Multiply(name="v4_gated_features")([fused, gate])
    gated = layers.Add(name="v4_gated_sum")([fused, scaled])

    attended = AttentionBlock("cbam", name="cbam_attention")(gated)
    tmd_features = layers.GlobalAveragePooling2D(name="tmd_gap")(attended)
    primary = layers.Dense(
        1024, activation="relu", kernel_regularizer=l2(config.l2_strength), name="tmd_fc1",
    )(tmd_features)
    primary = layers.Dropout(0.5, name="tmd_dropout")(primary)
    primary = layers.BatchNormalization(name="tmd_bn")(primary)
    primary = layers.Dense(128, activation="relu", name="tmd_fc2")(primary)
    tmd_output = layers.Dense(
        len(TMD_LABELS), activation="softmax", dtype="float32", name="tmd_output",
    )(primary)
    return Model(backbone.input, {"tmd_output": tmd_output, "artifact_output": artifact_output},
                 name=f"Candidate_{config.candidate}")


def build_model(config: DevelopmentConfig, *, weights: str | None = "imagenet"):
    if config.candidate in ("multiscale_gated", "artifact_gated"):
        return build_multiscale_gated_model(config, weights)

    import tensorflow as tf
    from tensorflow.keras import Model, layers
    from tensorflow.keras.regularizers import l2
    from scripts.pipeline.config import ARTIFACT_LABELS, TMD_LABELS
    from scripts.pipeline.models import AttentionBlock

    # Candidate B only changes the backbone.
    use_densenet121 = config.candidate == "densenet121"
    make_backbone = tf.keras.applications.DenseNet121 if use_densenet121 else tf.keras.applications.DenseNet201
    backbone = make_backbone(
        include_top=False, weights=weights,
        input_shape=(*config.input_hw, 3), pooling=None,
    )
    backbone.trainable = True
    conv5 = backbone.get_layer("conv5_block16_concat" if use_densenet121 else "conv5_block32_concat").output
    attended = AttentionBlock("cbam", name="cbam_attention")(conv5) if config.attention == "cbam" else conv5
    tmd_features = layers.GlobalAveragePooling2D(name="tmd_gap")(attended)
    primary = layers.Dense(
        1024, activation="relu", kernel_regularizer=l2(config.l2_strength), name="tmd_fc1",
    )(tmd_features)
    primary = layers.Dropout(0.5, name="tmd_dropout")(primary)
    primary = layers.BatchNormalization(name="tmd_bn")(primary)
    primary = layers.Dense(128, activation="relu", name="tmd_fc2")(primary)
    tmd_output = layers.Dense(
        len(TMD_LABELS), activation="softmax", dtype="float32", name="tmd_output",
    )(primary)

    if not config.multi_task:
        # TMJSequence(multi_task=False) yields an array target/sample weights.
        return Model(backbone.input, tmd_output, name=f"Controlled_{config.attention}_aux_off")

    # Unchanged pre-CBAM branch: never attach this to attended features.
    artifact_average = layers.GlobalAveragePooling2D(name="artifact_gap")(conv5)
    artifact_maximum = layers.GlobalMaxPooling2D(name="artifact_gmp")(conv5)
    artifact_features = layers.Concatenate(name="artifact_pooled_features")([artifact_average, artifact_maximum])
    auxiliary = layers.Dense(
        256, activation="relu", kernel_regularizer=l2(config.l2_strength), name="artifact_fc1",
    )(artifact_features)
    auxiliary = layers.Dropout(0.3, name="artifact_dropout")(auxiliary)
    artifact_output = layers.Dense(
        len(ARTIFACT_LABELS), activation="softmax", dtype="float32", name="artifact_output",
    )(auxiliary)
    return Model(backbone.input, {"tmd_output": tmd_output, "artifact_output": artifact_output},
                 name=f"Controlled_{config.attention}_aux_on")


def compile_model(model, config: DevelopmentConfig) -> None:
    import tensorflow as tf

    kwargs = {"optimizer": tf.keras.optimizers.Adam(config.learning_rate), "jit_compile": False}
    if config.multi_task:
        kwargs.update(
            loss={"tmd_output": "categorical_crossentropy", "artifact_output": "categorical_crossentropy"},
            loss_weights={"tmd_output": 1.0, "artifact_output": config.artifact_loss_weight},
            metrics={"tmd_output": [tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                                    tf.keras.metrics.CategoricalCrossentropy(name="primary_ce")],
                     "artifact_output": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]},
        )
    else:
        kwargs.update(loss="categorical_crossentropy", metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.CategoricalCrossentropy(name="primary_ce"),
        ])
    model.compile(**kwargs)
