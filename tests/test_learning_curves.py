from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.train_one_case_5fold import save_learning_curves


class LearningCurveTests(unittest.TestCase):
    def test_proposed_two_stage_learning_curve_is_created(self) -> None:
        history = pd.DataFrame(
            {
                "training_stage": [
                    "frozen_warmup",
                    "selective_fine_tuning",
                    "selective_fine_tuning",
                ],
                "loss": [2.0, 1.5, 1.2],
                "val_loss": [2.1, 1.6, 1.3],
                "tmd_output_loss": [0.7, 0.6, 0.5],
                "val_tmd_output_loss": [0.72, 0.63, 0.55],
                "artifact_output_loss": [1.2, 0.8, 0.5],
                "val_artifact_output_loss": [1.3, 0.9, 0.6],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curve.png"
            save_learning_curves(history, "proposed", output, "Test curve")
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
