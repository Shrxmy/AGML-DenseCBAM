"""Strict, stdlib-only fold binding for controlled development.

This audit reads encoded image bytes in train/validation/test FOR INTEGRITY ONLY.
It never decodes images or imports the training pipeline. Byte identity and declared
study groups do not establish patient independence or near-duplicate separation.
Filesystem checks reject symlinks before traversal/read, with no-follow opens where
available; they are not a guarantee against adversarial concurrent ancestor swaps.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
from pathlib import Path

# Keep in sync with scripts/pipeline/config.py without importing ML dependencies.
SPLITS = ("train", "validation", "test")
CLASSES = ("normal", "subluxation")
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
_REQUIRED = {"copied_path", "class_name", "fold_split", "content_sha256", "group_id"}
_FOLD = re.compile(r"fold_\d+")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def _stamp(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _check_path(path: Path, *, directory: bool = False) -> os.stat_result:
    """Check every ancestor without resolve(), which would hide symlinks."""
    for ancestor in reversed(path.parents):
        info = ancestor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Symlink/non-directory ancestor: {ancestor}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Symlink rejected: {path}")
    valid = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not valid:
        raise ValueError(f"Nonregular {'directory' if directory else 'file'}: {path}")
    return info


def _open_regular(path: Path):
    before = _check_path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stamp(opened) != _stamp(before):
            raise RuntimeError(f"File changed before integrity read: {path}")
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    return handle, opened


def _check_unchanged(path: Path, handle, before: os.stat_result) -> None:
    if (_stamp(os.fstat(handle.fileno())) != _stamp(before)
            or _stamp(_check_path(path)) != _stamp(before)):
        raise RuntimeError(f"File changed during integrity audit: {path}")


def _manifest_snapshot(path: Path) -> tuple[bytes, tuple]:
    handle, before = _open_regular(path)
    with handle:
        snapshot = handle.read()
        _check_unchanged(path, handle, before)
    return snapshot, _stamp(before)


def _hash_image(path: Path) -> str:
    handle, before = _open_regular(path)
    digest = hashlib.sha256()
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        _check_unchanged(path, handle, before)
    return digest.hexdigest()


def _copied_identity(value: str, fold_name: str) -> str:
    """Discard historical POSIX/Windows prefixes, never resolve manifest paths."""
    if not value.strip() or "\x00" in value:
        raise ValueError("Manifest copied_path is blank or contains NUL")
    parts = value.replace("\\", "/").split("/")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Manifest copied_path contains . or .. components")
    markers = [index for index, part in enumerate(parts) if _FOLD.fullmatch(part)]
    if len(markers) != 1 or parts[markers[0]] != fold_name:
        raise ValueError("Manifest copied_path must contain exactly the matching fold marker")
    suffix = parts[markers[0]:]
    if (len(suffix) < 4 or any(not part or ":" in part for part in suffix)
            or suffix[1] not in SPLITS or suffix[2] not in CLASSES
            or Path(suffix[-1]).suffix.lower() not in IMAGE_EXTENSIONS):
        raise ValueError("Invalid manifest copied_path split/class/image suffix")
    return "/".join(suffix)


def _parse_manifest(snapshot: bytes, fold_name: str) -> dict[str, dict]:
    try:
        reader = csv.reader(io.StringIO(snapshot.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader, [])
        if (not header or any(not field.strip() for field in header)
                or len(header) != len(set(header))):
            raise ValueError("Manifest header has blank or duplicate fields")
        if not _REQUIRED.issubset(header):
            raise ValueError(f"Manifest missing required fields: {sorted(_REQUIRED - set(header))}")
        records = {}
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise ValueError(f"Malformed/blank manifest row at record {line_number}")
            row = dict(zip(header, values))
            split, label = row["fold_split"], row["class_name"]
            if split not in SPLITS or label not in CLASSES:
                raise ValueError(f"Invalid manifest split/class at record {line_number}")
            if not _SHA256.fullmatch(row["content_sha256"]):
                raise ValueError(f"Manifest requires full SHA256 at record {line_number}")
            if not row["group_id"].strip():
                raise ValueError(f"Manifest requires nonblank group_id at record {line_number}")
            identity = _copied_identity(row["copied_path"], fold_name)
            parts = identity.split("/")
            if parts[1:3] != [split, label]:
                raise ValueError(f"Manifest copied_path split/class mismatch at record {line_number}")
            if identity in records:
                raise ValueError(f"Duplicate manifest copied_path identity: {identity}")
            row["content_sha256"] = row["content_sha256"].lower()
            row["group_id"] = row["group_id"].strip()
            records[identity] = row
        return records
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Malformed manifest CSV: {exc}") from exc


def _scan_images(directory: Path) -> list[Path]:
    """Scan without following links; complete scanning precedes image hashing."""
    _check_path(directory, directory=True)
    files = []
    with os.scandir(directory) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        path = directory / entry.name
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink rejected before image traversal/hash: {path}")
        if stat.S_ISDIR(info.st_mode):
            files.extend(_scan_images(path))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Nonregular image file: {path}")
            if "\\" in entry.name or ":" in entry.name:
                raise ValueError(f"Nonportable image filename: {path}")
            files.append(path)
    return sorted(files)


def audit_fold(fold: Path) -> tuple[list[dict], dict]:
    """Bind all indexed images to one exact parsed manifest-byte snapshot.

    Returns pandas-compatible rows (including test, for the caller to discard
    before evaluation) and JSON-compatible audit metadata. split_metadata uses
    {split: {count, class_counts}}; manifest_split_counts uses {split: count}.
    Raises ValueError for invalid data/paths, OSError for missing/unreadable paths,
    and RuntimeError when a file/manifest changes during the audit. No writes.
    """
    fold = Path(fold)
    # Preserve lexical components until checked: abspath/resolve could hide links.
    if ".." in fold.parts:
        raise ValueError("Fold path must not contain ..")
    if not fold.is_absolute():
        fold = Path.cwd() / fold
    if not _FOLD.fullmatch(fold.name):
        raise ValueError("Fold directory must have a fold_N name")
    _check_path(fold, directory=True)
    manifest = fold / "manifest.csv"
    snapshot, manifest_stamp = _manifest_snapshot(manifest)
    declared = _parse_manifest(snapshot, fold.name)

    rows = []
    indexed = {}
    counts = {split: {label: 0 for label in CLASSES} for split in SPLITS}
    for split in SPLITS:
        _check_path(fold / split, directory=True)
        for label_index, label in enumerate(CLASSES):
            directory = fold / split / label
            for path in _scan_images(directory):
                identity = f"{fold.name}/{path.relative_to(fold).as_posix()}"
                if identity in indexed:
                    raise ValueError(f"Duplicate indexed path identity: {identity}")
                indexed[identity] = path
                rows.append({"filepath": str(path), "split": split,
                             "class_name": label, "tmd_label": label_index})
                counts[split][label] += 1
    if any(count == 0 for classes in counts.values() for count in classes.values()):
        raise ValueError("Every train/validation/test split must contain both classes nonempty")
    if set(indexed) != set(declared):
        missing = sorted(set(indexed) - set(declared))
        extra = sorted(set(declared) - set(indexed))
        raise ValueError(f"Manifest/indexed path membership mismatch: missing={missing}, extra={extra}")

    content_labels = {}
    content_splits = {}
    group_splits = {}
    for identity, path in indexed.items():
        row = declared[identity]
        observed = _hash_image(path)
        if observed != row["content_sha256"]:
            raise ValueError(f"Observed bytes SHA256 mismatch against manifest: {identity}")
        split, label, group = row["fold_split"], row["class_name"], row["group_id"]
        if observed in content_labels and content_labels[observed] != label:
            raise ValueError("Content carries conflicting labels")
        if observed in content_splits and content_splits[observed] != split:
            raise ValueError("Content crosses train/validation/test splits")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Manifest group_id crosses train/validation/test splits")
        content_labels[observed] = label
        content_splits[observed] = split
        group_splits[group] = split

    final_snapshot, final_stamp = _manifest_snapshot(manifest)
    if final_snapshot != snapshot or final_stamp != manifest_stamp:
        raise RuntimeError(f"Manifest changed during integrity audit: {manifest}")
    split_metadata = {
        split: {"count": sum(classes.values()), "class_counts": classes}
        for split, classes in counts.items()
    }
    manifest_counts = {split: sum(row["fold_split"] == split for row in declared.values())
                       for split in SPLITS}
    return rows, {
        "manifest_sha256": hashlib.sha256(snapshot).hexdigest(),
        "split_metadata": split_metadata,
        "manifest_split_counts": manifest_counts,
        "byte_identity_only": True,
        "hash_scope": {
            "manifest": "complete raw bytes of the same manifest.csv snapshot parsed",
            "images": "complete encoded raw bytes of every indexed image",
            "splits": list(SPLITS),
            "purpose": "integrity only; no decoding, model evaluation or inference",
            "manifest_unchanged_after_audit": True,
            "patient_or_near_duplicate_independence_verified": False,
        },
    }
