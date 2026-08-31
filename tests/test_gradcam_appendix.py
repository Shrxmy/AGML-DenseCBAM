from __future__ import annotations

import unittest

import pandas as pd

from scripts.generate_gradcam_appendix import (
    select_balanced_samples,
    validate_heldout_predictions,
)


class GradcamAppendixSelectionTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_prediction_independent(self) -> None:
        predictions = pd.DataFrame(
            {
                "filepath": [f"/tmp/image_{index}.jpg" for index in range(12)],
                "y_true": [0] * 6 + [1] * 6,
                "y_pred": [1, 0, 1, 0, 1, 0] * 2,
            }
        )
        first = select_balanced_samples(predictions, 42, "clean", "fold_1", 2)
        changed_predictions = predictions.copy()
        changed_predictions["y_pred"] = 1 - changed_predictions["y_pred"]
        second = select_balanced_samples(changed_predictions, 42, "clean", "fold_1", 2)

        self.assertEqual(first["y_true"].value_counts().to_dict(), {0: 2, 1: 2})
        self.assertEqual(first["filepath"].tolist(), second["filepath"].tolist())
        self.assertEqual(first["selection_key"].tolist(), second["selection_key"].tolist())

    def test_rejects_non_test_prediction_paths(self) -> None:
        predictions = pd.DataFrame(
            {"filepath": ["/project/data_5_fold/fold_1/validation/normal/a.jpg"]}
        )
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_heldout_predictions(predictions, "fold_1")


if __name__ == "__main__":
    unittest.main()
