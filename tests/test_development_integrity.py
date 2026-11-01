"""Temporary raw-byte fixtures only: no real dataset, image decoder, or ML import."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.development import integrity


class DevelopmentIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fold = self.root / "fold_1"
        self.rows = []
        for split in integrity.SPLITS:
            for label in integrity.CLASSES:
                directory = self.fold / split / label
                directory.mkdir(parents=True)
                data = f"fake encoded bytes: {split}/{label}".encode()
                path = directory / f"{split}_{label}.jpg"
                path.write_bytes(data)
                self.rows.append({
                    "copied_path": f"data_5_fold/fold_1/{split}/{label}/{path.name}",
                    "class_name": label,
                    "fold_split": split,
                    "content_sha256": hashlib.sha256(data).hexdigest(),
                    "group_id": f"study-{split}-{label}",
                    "source_path": "historical source not opened.jpg",
                })
        self.fields = list(self.rows[0])
        self.manifest = self.fold / "manifest.csv"
        self.write_manifest()

    def write_manifest(self) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=self.fields)
        writer.writeheader()
        writer.writerows(self.rows)
        data = buffer.getvalue().encode()
        self.manifest.write_bytes(data)
        return data

    def image(self, index: int) -> Path:
        identity = integrity._copied_identity(self.rows[index]["copied_path"], self.fold.name)
        return self.root / identity

    def assert_invalid(self, message: str) -> None:
        with self.assertRaisesRegex(ValueError, message):
            integrity.audit_fold(self.fold)

    def add_image(self, split: str, label: str, name: str, data: bytes, group: str) -> None:
        path = self.fold / split / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.rows.append({
            "copied_path": f"fold_1/{split}/{label}/{name}", "class_name": label,
            "fold_split": split, "content_sha256": hashlib.sha256(data).hexdigest(),
            "group_id": group, "source_path": "ignored",
        })
        self.write_manifest()

    def symlink(self, target: Path, link: Path, directory: bool = False) -> None:
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")

    def test_valid_snapshot_api_and_metadata(self) -> None:
        raw = self.manifest.read_bytes()
        with patch.object(integrity, "_hash_image", wraps=integrity._hash_image) as hashed:
            rows, metadata = integrity.audit_fold(self.fold)
        self.assertEqual(hashed.call_count, 6)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(set(row), {"filepath", "split", "class_name", "tmd_label"})
            self.assertTrue(Path(row["filepath"]).is_absolute())
            self.assertEqual(row["tmd_label"], integrity.CLASSES.index(row["class_name"]))
        self.assertEqual(metadata["manifest_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(metadata["manifest_split_counts"], {split: 2 for split in integrity.SPLITS})
        self.assertEqual(metadata["split_metadata"], {
            split: {"count": 2, "class_counts": {label: 1 for label in integrity.CLASSES}}
            for split in integrity.SPLITS
        })
        self.assertTrue(metadata["byte_identity_only"])
        self.assertEqual(metadata["hash_scope"]["splits"], list(integrity.SPLITS))
        self.assertFalse(metadata["hash_scope"]["patient_or_near_duplicate_independence_verified"])

    def test_portable_historical_prefixes(self) -> None:
        prefixes = ("", "data_5_fold/", "/old/linux/data_5_fold/",
                    "C:\\old\\data_5_fold\\", "\\\\server\\share\\")
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                for row in self.rows:
                    suffix = integrity._copied_identity(row["copied_path"], "fold_1")
                    row["copied_path"] = prefix + (suffix.replace("/", "\\") if "\\" in prefix else suffix)
                self.write_manifest()
                self.assertEqual(len(integrity.audit_fold(self.fold)[0]), 6)

    def test_nested_regular_paths_and_nonimages(self) -> None:
        (self.fold / "train" / "normal" / "notes.txt").write_bytes(b"not indexed")
        self.add_image("train", "normal", "nested/deeper/extra.JPG", b"nested bytes", "nested-study")
        rows, meta = integrity.audit_fold(self.fold)
        self.assertEqual(len(rows), 7)
        self.assertEqual(meta["split_metadata"]["train"]["class_counts"]["normal"], 2)

    def test_stable_extensions_only(self) -> None:
        self.assertEqual(integrity.IMAGE_EXTENSIONS, {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        for index, extension in enumerate(sorted(integrity.IMAGE_EXTENSIONS)):
            self.add_image("train", "normal", f"extra{index}{extension.upper()}",
                           f"extra bytes {index}".encode(), f"extra-{index}")
        (self.fold / "train" / "normal" / "ignored.webp").write_bytes(b"ignored")
        self.assertEqual(len(integrity.audit_fold(self.fold)[0]), 12)

    def test_bom_hash_covers_exact_parsed_bytes(self) -> None:
        self.manifest.write_bytes(b"\xef\xbb\xbf" + self.manifest.read_bytes())
        raw = self.manifest.read_bytes()
        self.assertEqual(integrity.audit_fold(self.fold)[1]["manifest_sha256"], hashlib.sha256(raw).hexdigest())

    def test_same_count_wrong_membership_fails_before_hash(self) -> None:
        self.rows[0]["copied_path"] = "fold_1/train/normal/stale.jpg"
        self.write_manifest()
        with patch.object(integrity, "_hash_image", side_effect=AssertionError("must not hash")):
            self.assert_invalid("membership mismatch")

    def test_extra_indexed_image(self) -> None:
        (self.fold / "train" / "normal" / "unlisted.jpg").write_bytes(b"extra")
        self.assert_invalid("membership mismatch")

    def test_extra_manifest_row(self) -> None:
        self.rows.append(dict(self.rows[0], copied_path="fold_1/train/normal/ghost.jpg"))
        self.write_manifest()
        self.assert_invalid("membership mismatch")

    def test_wrong_class_and_split_fields(self) -> None:
        for field, value in (("class_name", "subluxation"), ("fold_split", "validation")):
            with self.subTest(field=field):
                original = self.rows[0][field]
                self.rows[0][field] = value
                self.write_manifest()
                self.assert_invalid("split/class mismatch")
                self.rows[0][field] = original

    def test_wrong_fold_missing_marker_and_ambiguous_marker(self) -> None:
        for value in ("fold_2/train/normal/a.jpg", "train/normal/a.jpg",
                      "fold_1/old/fold_1/train/normal/a.jpg", "fold_2/old/fold_1/train/normal/a.jpg"):
            with self.subTest(path=value):
                self.rows[0]["copied_path"] = value
                self.write_manifest()
                self.assert_invalid("fold marker")

    def test_dot_components_rejected_not_normalized(self) -> None:
        for value in ("old/../fold_1/train/normal/a.jpg", "old/./fold_1/train/normal/a.jpg",
                      "fold_1/train/normal/sub/../a.jpg", "fold_1\\train\\normal\\.\\a.jpg"):
            with self.subTest(path=value):
                self.rows[0]["copied_path"] = value
                self.write_manifest()
                self.assert_invalid("components")

    def test_invalid_path_suffixes(self) -> None:
        for value in ("", " ", "fold_1/train/normal/", "fold_1/train/normal/a.webp",
                      "fold_1/evaluation/normal/a.jpg", "fold_1/train/unknown/a.jpg",
                      "fold_1/train/normal//a.jpg", "fold_1/train/normal/a.jpg\x00"):
            with self.subTest(path=value):
                self.rows[0]["copied_path"] = value
                self.write_manifest()
                self.assert_invalid("copied_path")

    def test_duplicate_normalized_copied_paths(self) -> None:
        duplicate = dict(self.rows[0])
        duplicate["copied_path"] = "Z:\\historical\\" + integrity._copied_identity(
            duplicate["copied_path"], "fold_1").replace("/", "\\")
        self.rows.append(duplicate)
        self.write_manifest()
        self.assert_invalid("Duplicate manifest copied_path")

    def test_blank_or_malformed_full_sha(self) -> None:
        for digest in ("", " ", "a" * 63, "a" * 65, "z" * 64, " " + "a" * 64, "a" * 16):
            with self.subTest(digest=digest):
                self.rows[0]["content_sha256"] = digest
                self.write_manifest()
                self.assert_invalid("full SHA256")

    def test_uppercase_digest_is_valid(self) -> None:
        self.rows[0]["content_sha256"] = self.rows[0]["content_sha256"].upper()
        self.write_manifest()
        self.assertEqual(len(integrity.audit_fold(self.fold)[0]), 6)

    def test_blank_group(self) -> None:
        for group in ("", " \t "):
            self.rows[0]["group_id"] = group
            self.write_manifest()
            self.assert_invalid("nonblank group_id")

    def test_unknown_or_blank_split_class(self) -> None:
        for field in ("fold_split", "class_name"):
            for value in ("", " ", "unknown"):
                with self.subTest(field=field, value=value):
                    original = self.rows[0][field]
                    self.rows[0][field] = value
                    self.write_manifest()
                    self.assert_invalid("Invalid manifest split/class")
                    self.rows[0][field] = original

    def test_observed_bytes_must_match_declared_sha(self) -> None:
        self.image(0).write_bytes(b"changed fake bytes")
        self.assert_invalid("Observed bytes SHA256 mismatch")

    def test_conflicting_content_labels_same_split(self) -> None:
        data = self.image(0).read_bytes()
        self.image(1).write_bytes(data)
        self.rows[1]["content_sha256"] = hashlib.sha256(data).hexdigest()
        self.write_manifest()
        self.assert_invalid("conflicting labels")

    def test_content_cross_split_same_label(self) -> None:
        data = self.image(0).read_bytes()
        self.image(2).write_bytes(data)
        self.rows[2]["content_sha256"] = hashlib.sha256(data).hexdigest()
        self.write_manifest()
        self.assert_invalid("Content crosses")

    def test_group_cross_split(self) -> None:
        self.rows[2]["group_id"] = self.rows[0]["group_id"]
        self.write_manifest()
        self.assert_invalid("group_id crosses")

    def test_group_whitespace_does_not_hide_leakage(self) -> None:
        self.rows[2]["group_id"] = " " + self.rows[0]["group_id"] + " "
        self.write_manifest()
        self.assert_invalid("group_id crosses")

    def test_group_and_same_label_content_within_split_allowed(self) -> None:
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        data = self.image(0).read_bytes()
        self.add_image("train", "normal", "duplicate.jpg", data, self.rows[0]["group_id"])
        self.assertEqual(len(integrity.audit_fold(self.fold)[0]), 7)

    def test_missing_class_in_each_split(self) -> None:
        for index in (0, 2, 4):
            with self.subTest(split=self.rows[index]["fold_split"]):
                path = self.image(index)
                raw = path.read_bytes()
                path.unlink()
                self.assert_invalid("both classes nonempty")
                path.write_bytes(raw)

    def test_missing_split_directory(self) -> None:
        (self.fold / "validation").rename(self.fold / "missing_validation")
        with self.assertRaises(FileNotFoundError):
            integrity.audit_fold(self.fold)

    def test_duplicate_csv_fields_including_unused_fields(self) -> None:
        raw = self.manifest.read_bytes()
        for field in ("copied_path", "source_path"):
            header, body = raw.split(b"\r\n", 1)
            self.manifest.write_bytes(header + b"," + field.encode() + b"\r\n" + body)
            self.assert_invalid("duplicate fields")

    def test_missing_required_csv_field(self) -> None:
        self.fields.remove("class_name")
        for row in self.rows:
            del row["class_name"]
        self.write_manifest()
        self.assert_invalid("missing required fields")

    def test_malformed_csv_rows(self) -> None:
        valid = self.manifest.read_bytes()
        for tail in (b"\r\n", b"one,two\r\n", b",,,,,,extra\r\n", b'"unterminated'):
            with self.subTest(tail=tail):
                self.manifest.write_bytes(valid + tail)
                self.assert_invalid("Malformed")

    def test_invalid_utf8_and_empty_manifest(self) -> None:
        for raw in (b"\xff", b"", b",,,,,\n"):
            with self.subTest(raw=raw):
                self.manifest.write_bytes(raw)
                self.assert_invalid("manifest|Manifest")

    def test_manifest_mutation_during_image_hashing(self) -> None:
        original = integrity._hash_image
        changed = False

        def mutate(path: Path) -> str:
            nonlocal changed
            if not changed:
                changed = True
                self.manifest.write_bytes(self.manifest.read_bytes().replace(b"study-train", b"other-train"))
            return original(path)

        with patch.object(integrity, "_hash_image", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "Manifest changed"):
                integrity.audit_fold(self.fold)

    def test_manifest_replacement_with_same_bytes_detected(self) -> None:
        original = integrity._hash_image
        changed = False

        def replace(path: Path) -> str:
            nonlocal changed
            if not changed:
                changed = True
                replacement = self.root / "replacement.csv"
                replacement.write_bytes(self.manifest.read_bytes())
                os.replace(replacement, self.manifest)
            return original(path)

        with patch.object(integrity, "_hash_image", side_effect=replace):
            with self.assertRaisesRegex(RuntimeError, "Manifest changed"):
                integrity.audit_fold(self.fold)

    def test_symlink_fold_split_class_nested_and_image_before_hash(self) -> None:
        targets = (self.fold, self.fold / "test", self.fold / "test" / "normal",
                   self.image(4))
        for index, target in enumerate(targets):
            with self.subTest(target=target):
                actual = self.root / f"real-{index}"
                target.rename(actual)
                self.symlink(actual, target, directory=actual.is_dir())
                with patch.object(integrity, "_hash_image", side_effect=AssertionError("must not hash")):
                    self.assert_invalid("Symlink")
                target.unlink()
                actual.rename(target)
        nested = self.fold / "test" / "subluxation" / "nested"
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "escape.jpg").write_bytes(b"not read")
        self.symlink(outside, nested, directory=True)
        with patch.object(integrity, "_hash_image", side_effect=AssertionError("must not hash")):
            self.assert_invalid("Symlink")

    def test_symlink_alias_inside_class_rejected(self) -> None:
        self.symlink(self.image(0), self.image(0).with_name("alias.jpg"))
        with patch.object(integrity, "_hash_image", side_effect=AssertionError("must not hash")):
            self.assert_invalid("Symlink")

    def test_dangling_image_and_manifest_symlinks(self) -> None:
        link = self.fold / "test" / "normal" / "dangling.jpg"
        self.symlink(self.root / "absent", link)
        self.assert_invalid("Symlink")
        link.unlink()
        self.manifest.unlink()
        self.symlink(self.root / "absent.csv", self.manifest)
        self.assert_invalid("Symlink")

    def test_symlink_fold_ancestor_rejected(self) -> None:
        link = self.root / "alias"
        self.symlink(self.root, link, directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            integrity.audit_fold(link / "fold_1")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation unavailable")
    def test_nonregular_image_fifo_rejected_before_open(self) -> None:
        os.mkfifo(self.fold / "test" / "normal" / "pipe.jpg")
        with patch.object(integrity, "_hash_image", side_effect=AssertionError("must not hash")):
            self.assert_invalid("Nonregular image")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation unavailable")
    def test_nonregular_manifest_fifo_rejected(self) -> None:
        self.manifest.unlink()
        os.mkfifo(self.manifest)
        self.assert_invalid("Nonregular")

    def test_relative_fold_path(self) -> None:
        with patch.object(Path, "cwd", return_value=self.root):
            self.assertEqual(len(integrity.audit_fold(Path("fold_1"))[0]), 6)

    def test_invalid_fold_name_and_parent_components(self) -> None:
        for path in (self.root / "unexpected", self.root / "other" / ".." / "fold_1"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "fold_N|contain .."):
                    integrity.audit_fold(path)

    def test_import_is_stdlib_only_under_python_no_site(self) -> None:
        project = Path(__file__).absolute().parents[1]
        code = (
            "import sys; from scripts.development.integrity import audit_fold; "
            "assert not any(name.split('.')[0] in "
            "{'tensorflow','keras','cv2','numpy','pandas'} for name in sys.modules); "
            "assert callable(audit_fold)"
        )
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=project,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
