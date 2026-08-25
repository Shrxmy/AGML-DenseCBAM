from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_chapter4 import (
    CALIBRATION_METRICS,
    CASES,
    CORRUPTION_METRICS,
    EFFICIENCY_METRICS,
    PRIMARY_METRICS,
    load_cases,
)


class Chapter4InputValidationTests(unittest.TestCase):
    def _write_cases(self, root: Path, evaluation_split: str | None) -> None:
        for _, (model_type, scenario) in CASES.items():
            rows = []
            for fold_index in range(1, 6):
                row = {
                    "fold": f"fold_{fold_index}",
                    "model_type": model_type,
                    "scenario": scenario,
                    **{metric: 0.5 for metric in PRIMARY_METRICS},
                    **{metric: 1.0 for metric in EFFICIENCY_METRICS},
                    **{metric: 0.1 for metric in CALIBRATION_METRICS},
                    **{metric: 0.5 for metric in CORRUPTION_METRICS},
                }
                if evaluation_split is not None:
                    row["evaluation_split"] = evaluation_split
                    row["run_config_sha256"] = f"{model_type}-{scenario}"
                rows.append(row)
            case_dir = root / f"{model_type}_{scenario}"
            case_dir.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(case_dir / "all_fold_results.csv", index=False)

    def test_rejects_validation_only_development_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cases(root, evaluation_split="validation")
            with self.assertRaisesRegex(ValueError, "development/validation"):
                load_cases(root, expected_folds=5)

    def test_accepts_explicit_test_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cases(root, evaluation_split="test")
            combined = load_cases(root, expected_folds=5)
            self.assertEqual(len(combined), 20)


if __name__ == "__main__":
    unittest.main()
