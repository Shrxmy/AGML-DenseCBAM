from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from make_5fold_dataset import (
    create_folds,
    prepare_unique_items,
    scan_source_images,
)


class FoldGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        for split in ["train", "validation", "test"]:
            for class_name in ["normal", "subluxation"]:
                (self.data_root / split / class_name).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_balanced_images(self, per_class: int = 20) -> None:
        for class_name in ["normal", "subluxation"]:
            for index in range(per_class):
                split = ["train", "validation", "test"][index % 3]
                path = self.data_root / split / class_name / f"{class_name}_{index}.jpg"
                path.write_bytes(f"unique-{class_name}-{index}".encode())

    def test_conflicting_exact_image_stops_by_default(self) -> None:
        self.add_balanced_images()
        conflict = self.data_root / "test" / "subluxation" / "conflict.jpg"
        conflict.write_bytes(b"unique-normal-0")
        images = scan_source_images(self.data_root)

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_unique_items(
                images,
                output_root=self.root / "folds",
                input_root=self.data_root,
                conflict_policy="error",
                group_mapping={},
            )

    def test_patient_groups_remain_in_one_split(self) -> None:
        self.add_balanced_images()
        images = scan_source_images(self.data_root)
        group_mapping = {
            str(image.path).replace("\\", "/"): f"patient_{image.path.stem.split('_')[-1]}"
            for image in images
        }
        items = prepare_unique_items(
            images,
            output_root=self.root / "grouped_folds",
            input_root=self.data_root,
            conflict_policy="error",
            group_mapping=group_mapping,
        )
        create_folds(items, self.root / "grouped_folds", n_splits=5, val_size=0.15, random_state=42)

        for manifest in sorted((self.root / "grouped_folds").glob("fold_*/manifest.csv")):
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            group_splits = {}
            for row in rows:
                group_splits.setdefault(row["group_id"], set()).add(row["fold_split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_exact_duplicates_are_deduplicated_and_do_not_leak(self) -> None:
        self.add_balanced_images()
        duplicate = self.data_root / "test" / "normal" / "duplicate.jpg"
        duplicate.write_bytes(b"unique-normal-0")
        images = scan_source_images(self.data_root)
        items = prepare_unique_items(
            images,
            output_root=self.root / "folds",
            input_root=self.data_root,
            conflict_policy="error",
            group_mapping={},
        )
        self.assertEqual(len(items), 40)

        create_folds(items, self.root / "folds", n_splits=5, val_size=0.15, random_state=42)
        test_hashes = []
        for manifest in sorted((self.root / "folds").glob("fold_*/manifest.csv")):
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            split_hashes = {
                split: {row["content_sha256"] for row in rows if row["fold_split"] == split}
                for split in ["train", "validation", "test"]
            }
            self.assertTrue(split_hashes["train"].isdisjoint(split_hashes["validation"]))
            self.assertTrue(split_hashes["train"].isdisjoint(split_hashes["test"]))
            self.assertTrue(split_hashes["validation"].isdisjoint(split_hashes["test"]))
            test_hashes.append(split_hashes["test"])

        self.assertEqual(sum(map(len, test_hashes)), 40)
        self.assertEqual(len(set().union(*test_hashes)), 40)


if __name__ == "__main__":
    unittest.main()
