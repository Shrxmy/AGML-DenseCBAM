from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from scripts.train_one_case_5fold import (
    add_metal_streak,
    apply_artifact,
    apply_conservative_augmentation,
    configure_selective_fine_tuning,
)


class ArtifactProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        gradient = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
        self.image = np.repeat(gradient[..., None], 3, axis=-1)

    def test_v2_metal_streak_is_visible_for_multiple_seeds(self) -> None:
        for seed in range(20):
            transformed = add_metal_streak(
                self.image,
                np.random.default_rng(seed),
                num_streaks=1 + seed % 2,
            )
            self.assertEqual(transformed.shape, self.image.shape)
            self.assertFalse(np.array_equal(transformed, self.image))
            self.assertGreater(np.mean(np.abs(transformed.astype(float) - self.image)), 0.1)

    def test_artifacts_are_deterministic_for_a_fixed_seed(self) -> None:
        for artifact_label in [0, 1, 2, 3]:
            first = apply_artifact(self.image, artifact_label, np.random.default_rng(42))
            second = apply_artifact(self.image, artifact_label, np.random.default_rng(42))
            np.testing.assert_array_equal(first, second)

    def test_v3_augmentation_is_deterministic_and_shape_preserving(self) -> None:
        first = apply_conservative_augmentation(self.image, np.random.default_rng(42))
        second = apply_conservative_augmentation(self.image, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, self.image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertFalse(np.array_equal(first, self.image))

    def test_v3_selective_fine_tuning_freezes_backbone_batch_norm(self) -> None:
        inputs = tf.keras.Input((4,), name="input")
        side_head = tf.keras.layers.Dense(4, name="benchmark_self_attention")(inputs)
        x = tf.keras.layers.Dense(4, name="early_backbone")(inputs)
        x = tf.keras.layers.Dense(4, name="conv5_block1_0_bn")(x)
        x = tf.keras.layers.BatchNormalization(name="conv5_batch_norm")(x)
        x = tf.keras.layers.ReLU(name="conv5_block32_concat")(x)
        x = tf.keras.layers.Add(name="benchmark_fused_features")([x, side_head])
        outputs = tf.keras.layers.Dense(2, name="tmd_output")(x)
        model = tf.keras.Model(inputs, outputs)
        model._agml_backbone_layer_names = (
            "input",
            "early_backbone",
            "conv5_block1_0_bn",
            "conv5_batch_norm",
            "conv5_block32_concat",
        )
        config = SimpleNamespace(
            freeze_backbone=False,
            fine_tune_from_layer="conv5_block1_0_bn",
            freeze_batch_norm=True,
        )

        counts = configure_selective_fine_tuning(model, config)

        self.assertFalse(model.get_layer("early_backbone").trainable)
        self.assertTrue(model.get_layer("conv5_block1_0_bn").trainable)
        self.assertFalse(model.get_layer("conv5_batch_norm").trainable)
        self.assertTrue(model.get_layer("conv5_block32_concat").trainable)
        self.assertTrue(model.get_layer("benchmark_self_attention").trainable)
        self.assertTrue(model.get_layer("tmd_output").trainable)
        self.assertEqual(counts["unfrozen_backbone_layers"], 2)
        self.assertEqual(counts["frozen_batch_norm_layers"], 1)


if __name__ == "__main__":
    unittest.main()
