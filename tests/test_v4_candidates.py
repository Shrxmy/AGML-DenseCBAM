"""V4 candidates: structure checks, and the default build must stay unchanged."""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.development.config import DevelopmentConfig, validate_config
from scripts.development.models import build_model, compile_model

CONFIG = DevelopmentConfig(folds_root=Path("data_5_fold"), output_dir=Path("results/development_controlled/x"),
                           image_size=224, image_height=448, recipe="v3", batch_size=2)


def build(candidate: str):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    config = dataclasses.replace(CONFIG, candidate=candidate)
    model = build_model(config, weights=None)
    compile_model(model, config)
    return model, config


class CandidateTests(unittest.TestCase):
    def test_default_has_no_candidate_layers_or_fingerprint_field(self) -> None:
        model, config = build("none")
        self.assertFalse([layer.name for layer in model.layers if layer.name.startswith("v4_")])
        self.assertIsNotNone(model.get_layer("conv5_block32_concat"))
        self.assertNotIn("candidate", config.payload())

    def test_fingerprint_changes_with_candidate(self) -> None:
        self.assertEqual(dataclasses.replace(CONFIG, candidate="densenet121").payload()["candidate"], "densenet121")

    def test_candidates_need_cbam_and_auxiliary_head(self) -> None:
        for attention, auxiliary in (("none", "on"), ("cbam", "off")):
            with self.assertRaises(ValueError):
                validate_config(dataclasses.replace(CONFIG, candidate="densenet121",
                                                     attention=attention, auxiliary_head=auxiliary))

    def test_densenet121_is_smaller_with_same_outputs(self) -> None:
        small, _ = build("densenet121")
        small_params = small.count_params()
        self.assertEqual(sorted(small.output), ["artifact_output", "tmd_output"])
        self.assertEqual(small.get_layer("cbam_attention").input.shape[1:3], (14, 7))
        default, _ = build("none")
        self.assertLess(small_params, default.count_params())

    def test_multiscale_gate_is_trained_and_aligned(self) -> None:
        model, _ = build("multiscale_gated")
        self.assertEqual(model.get_layer("v4_fused_features").output.shape[1:3], (14, 7))
        self.assertEqual(model.get_layer("cbam_attention").input.shape[1:], (14, 7, 1024))

        x = tf.constant(np.random.default_rng(0).random((2, 448, 224, 3)), dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(x, training=False)["tmd_output"][:, 1])
        names = ("v4_artifact_gate", "v4_pool3_projection", "v4_fusion_conv", "artifact_fc1")
        weights = [model.get_layer(name).trainable_weights[0] for name in names]
        for name, gradient in zip(names, tape.gradient(loss, weights)):
            self.assertIsNotNone(gradient, name)

    def test_artifact_gated_has_gate_and_no_pool3_path(self) -> None:
        model, _ = build("artifact_gated")
        names = [layer.name for layer in model.layers]
        self.assertFalse([name for name in names if "pool3_" in name and name.startswith("v4_")])
        self.assertEqual(model.get_layer("cbam_attention").input.shape[1:], (14, 7, 1024))

        x = tf.constant(np.random.default_rng(0).random((2, 448, 224, 3)), dtype=tf.float32)
        gate = model.get_layer("v4_artifact_gate").trainable_weights[0]
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(x, training=False)["tmd_output"][:, 1])
        self.assertIsNotNone(tape.gradient(loss, gate))


if __name__ == "__main__":
    unittest.main()
