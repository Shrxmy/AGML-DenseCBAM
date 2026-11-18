from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.gradcam_app import AUTO_FOLD, Record, checkpoint_path, choose_fold, data_note, load_records


class GradcamAppTests(unittest.TestCase):
    def test_checkpoint_path_matches_saved_layout(self) -> None:
        path = checkpoint_path(Path("/r"), "V3 follow-up", "C4 Proposed, artifact mix", "fold_2")
        self.assertEqual(path, Path("/r/results/final_v3/proposed_artifact_mix/fold_2_proposed_artifact_mix_best.keras"))
        path = checkpoint_path(Path("/r"), "Final V2 (primary)", "C1 Benchmark, clean", "fold_1")
        self.assertEqual(path, Path("/r/results/final_v2/benchmark_clean/fold_1_benchmark_clean_best.keras"))

    def test_records_join_manifests_by_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for fold, split in (("fold_1", "train"), ("fold_2", "test")):
                folder = Path(temp) / fold
                folder.mkdir()
                (folder / "manifest.csv").write_text(f"content_sha256,class_name,fold_split\nabc,normal,{split}\n")
            records = load_records(Path(temp))
        self.assertEqual(records["abc"], Record("normal", {"fold_1": "train", "fold_2": "test"}))

    def test_auto_fold_uses_the_held_out_model(self) -> None:
        record = Record("normal", {"fold_1": "train", "fold_2": "validation", "fold_3": "test"})
        self.assertEqual(choose_fold(AUTO_FOLD, record), "fold_3")
        self.assertEqual(choose_fold("fold_1", record), "fold_1")
        self.assertEqual(choose_fold(AUTO_FOLD, None), "fold_1")

    def test_note_warns_unless_the_image_was_held_out(self) -> None:
        record = Record("subluxation", {"fold_1": "train", "fold_2": "validation", "fold_3": "test"})
        self.assertIn("fair test", data_note("fold_3", record))
        self.assertNotIn("Warning", data_note("fold_3", record))
        self.assertIn("Warning", data_note("fold_1", record))
        self.assertIn("Warning", data_note("fold_2", record))
        self.assertIn("Recorded label: Subluxation", data_note("fold_3", record))
        self.assertIn("not in the study folds", data_note("fold_1", None))


if __name__ == "__main__":
    unittest.main()
