from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_learning_curves import generate_for_results_root
from scripts.train_one_case_5fold import save_learning_curves


class LearningCurveTests(unittest.TestCase):
    def test_proposed_learning_curve_is_created(self) -> None:
        history = pd.DataFrame(
            {
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

    def test_separate_output_root_preserves_source_directory(self) -> None:
        history = pd.DataFrame(
            {
                "loss": [2.0, 1.5],
                "val_loss": [2.1, 1.6],
                "tmd_output_loss": [0.7, 0.6],
                "val_tmd_output_loss": [0.72, 0.63],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "source" / "benchmark_clean"
            case_dir.mkdir(parents=True)
            history.to_csv(case_dir / "fold_1_history.csv", index=False)
            outputs = generate_for_results_root(root / "source", root / "generated")
            expected = root / "generated" / "benchmark_clean" / "fold_1_learning_curves.png"
            self.assertEqual(outputs, [expected])
            self.assertTrue(expected.exists())
            self.assertFalse((case_dir / "fold_1_learning_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
