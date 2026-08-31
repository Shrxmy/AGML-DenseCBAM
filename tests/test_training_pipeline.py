from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import scripts.train_one_case_5fold as training
from scripts.pipeline.provenance import training_source_sha256


class TrainingPipelineStructureTests(unittest.TestCase):
    def test_entrypoint_keeps_the_public_training_interface(self) -> None:
        expected_names = {
            "RunConfig",
            "TMJSequence",
            "AttentionBlock",
            "add_metal_streak",
            "apply_artifact",
            "build_benchmark_model",
            "build_proposed_model",
            "collect_predictions",
            "run_one_fold",
            "run_case",
            "parse_args",
        }
        missing = sorted(name for name in expected_names if not hasattr(training, name))
        self.assertEqual(missing, [])

    def test_training_fingerprint_covers_pipeline_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scripts_dir = Path(temporary)
            pipeline_dir = scripts_dir / "pipeline"
            pipeline_dir.mkdir()
            (scripts_dir / "train_one_case_5fold.py").write_text("entry point\n")
            module = pipeline_dir / "models.py"
            module.write_text("first version\n")

            first = training_source_sha256(scripts_dir)
            module.write_text("second version\n")
            second = training_source_sha256(scripts_dir)

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
