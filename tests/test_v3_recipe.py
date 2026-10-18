"""V3 recipe: aspect-preserving input, torch preprocessing, mild augmentation, TTA."""
from __future__ import annotations

import dataclasses
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.pipeline.artifacts import densenet_torch_preprocess
from scripts.pipeline.config import RunConfig, effective_run_config_sha256
from scripts.pipeline.data import TMJSequence
from scripts.pipeline.evaluation import collect_predictions
from scripts.pipeline.provenance import sha256_json


class V3RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        rows = []
        for index in range(6):
            path = Path(cls.tmp.name) / f"img_{index}.jpg"
            # Portrait source like the real crops (W < H).
            cv2.imwrite(str(path), rng.integers(0, 256, (140, 70, 3), dtype=np.uint8))
            rows.append({"filepath": str(path), "tmd_label": index % 2, "split": "train"})
        cls.df = pd.DataFrame(rows)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def sequence(self, recipe: str, training: bool, size=(32, 32)) -> TMJSequence:
        return TMJSequence(self.df, size, 3, True, "artifact_mix", training, 42, recipe=recipe)

    def test_v2_batches_unchanged_by_cache(self) -> None:
        sequence = self.sequence("v2", training=False)
        first_x, _ = sequence[0]
        second_x, _ = sequence[0]  # Served from the cache.
        fresh_x, _ = self.sequence("v2", training=False)[0]
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_x, fresh_x)
        self.assertGreaterEqual(first_x.min(), -1.0)
        self.assertLessEqual(first_x.max(), 1.0)

    def test_v3_input_is_height_by_width(self) -> None:
        x, _ = self.sequence("v3", training=False, size=(64, 32))[0]
        self.assertEqual(x.shape, (3, 64, 32, 3))

    def test_torch_preprocessing_matches_keras_densenet(self) -> None:
        image = np.random.default_rng(1).integers(0, 256, (8, 4, 3)).astype(np.uint8)
        expected = tf.keras.applications.densenet.preprocess_input(image.astype(np.float32))
        np.testing.assert_allclose(densenet_torch_preprocess(image), expected, rtol=1e-5, atol=1e-5)

    def test_v3_augmentation_is_deterministic_and_keeps_artifact_labels(self) -> None:
        first_x, first_y = self.sequence("v3", training=True)[0]
        second_x, second_y = self.sequence("v3", training=True)[0]
        v2_x, v2_y = self.sequence("v2", training=True)[0]
        self.assertFalse(np.allclose(first_x, densenet_torch_preprocess((v2_x + 1) * 127.5)))
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y["artifact_output"], second_y["artifact_output"])
        np.testing.assert_array_equal(first_y["artifact_output"], v2_y["artifact_output"])

    def test_tta_averages_original_and_flipped_predictions(self) -> None:
        sequence = self.sequence("v3", training=False)

        def model(x, training=False):
            # Prediction depends on left/right asymmetry so the flip matters.
            score = 1 / (1 + np.exp(-(x[:, :, 0, 0].mean(axis=1) - x[:, :, -1, 0].mean(axis=1))))
            tmd = np.stack([1 - score, score], axis=1)
            return {"tmd_output": tmd, "artifact_output": np.full((len(x), 4), 0.25)}

        compiled = types.SimpleNamespace(predict_on_batch=model)  # V3 uses compiled inference.
        _, _, confidence, *_ = collect_predictions(compiled, sequence, multi_task=True)
        np.testing.assert_allclose(confidence, 0.5, atol=1e-6)

    def test_v2_run_config_fingerprint_ignores_recipe_field(self) -> None:
        config = RunConfig(
            folds_root=Path("data_5_fold"), output_dir=Path("results/x"), model_type="proposed",
            scenario="artifact_mix", image_size=(224, 224), batch_size=8, epochs=50,
            learning_rate=1e-4, l2_strength=1e-2, random_state=42, freeze_backbone=False,
            tmd_loss_weight=1.0, artifact_loss_weight=0.3, fold_limit=None, single_fold=None,
            verify_integrity=True, mixed_precision=True, class_weighting=True,
        )
        legacy = dataclasses.asdict(config)
        legacy.pop("recipe")
        legacy.update(folds_root=str(config.folds_root.resolve()), output_dir=str(config.output_dir.resolve()),
                      image_size=[224, 224], run_config_sha256=None)
        self.assertEqual(effective_run_config_sha256(config), sha256_json(legacy))
        v3 = dataclasses.replace(config, recipe="v3")
        self.assertNotEqual(effective_run_config_sha256(v3), effective_run_config_sha256(config))


    def test_development_v2_payload_omits_new_fields_and_v3_requires_height(self) -> None:
        from scripts.development.config import DevelopmentConfig, validate_config

        payload = DevelopmentConfig(Path("data_5_fold"), Path("results/development_controlled/x")).payload()
        self.assertNotIn("recipe", payload)
        self.assertNotIn("image_height", payload)
        v3 = DevelopmentConfig(Path("data_5_fold"), Path("results/development_controlled/x"), recipe="v3")
        with self.assertRaises(ValueError):
            validate_config(v3)
        validate_config(dataclasses.replace(v3, image_height=448))
        self.assertEqual(dataclasses.replace(v3, image_height=448).input_hw, (448, 224))

if __name__ == "__main__":
    unittest.main()
