from __future__ import annotations

import unittest

import numpy as np

from scripts.train_one_case_5fold import add_metal_streak, apply_artifact


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


if __name__ == "__main__":
    unittest.main()
