#!/usr/bin/env python
"""Create leakage-resistant stratified folds for the TMJ image dataset.

Exact duplicate images are identified by SHA-256 before splitting. Same-label
copies are represented once so identical pixels cannot occur in train and test.
Conflicting-label duplicate groups are reported and, by default, stop fold
creation rather than being resolved silently.

Patient/study-level separation still requires patient metadata. When a groups
CSV is supplied, all rows sharing a group_id are assigned to the same split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["normal", "subluxation"]
SOURCE_SPLITS = ["train", "validation", "test"]


@dataclass(frozen=True)
class SourceImage:
    path: Path
    class_name: str
    label: int
    original_split: str
    content_sha256: str


@dataclass(frozen=True)
class Item:
    """One unique image-content sample used by cross-validation."""

    src_path: Path
    class_name: str
    label: int
    original_split: str
    content_sha256: str
    duplicate_paths: Tuple[str, ...]
    group_id: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def scan_source_images(input_root: Path) -> List[SourceImage]:
    images: List[SourceImage] = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for split_name in SOURCE_SPLITS:
        split_dir = input_root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing source split folder: {split_dir}")

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
                    images.append(
                        SourceImage(
                            path=file_path,
                            class_name=class_name,
                            label=class_to_idx[class_name],
                            original_split=split_name,
                            content_sha256=sha256_file(file_path),
                        )
                    )

    if not images:
        raise ValueError(f"No images found under {input_root}")
    return images


def load_group_mapping(groups_csv: Path | None, input_root: Path) -> Dict[str, str]:
    """Load optional source_path -> patient/study group_id mapping."""
    if groups_csv is None:
        return {}

    mapping: Dict[str, str] = {}
    with groups_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"source_path", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{groups_csv} must contain columns: source_path, group_id")
        for row in reader:
            source = normalized_path(row["source_path"].strip())
            group_id = row["group_id"].strip()
            if not source or not group_id:
                raise ValueError(f"Blank source_path/group_id in {groups_csv}: {row}")
            mapping[source] = group_id
            source_path = Path(source)
            try:
                mapping[normalized_path(source_path.relative_to(input_root))] = group_id
            except ValueError:
                pass
            if source_path.parts and source_path.parts[0] == input_root.name:
                mapping[normalized_path(Path(*source_path.parts[1:]))] = group_id
    return mapping


def resolve_group_id(
    members: Sequence[SourceImage],
    group_mapping: Dict[str, str],
    input_root: Path,
) -> str:
    if not group_mapping:
        return members[0].content_sha256

    group_ids = set()
    missing = []
    for member in members:
        candidates = [normalized_path(member.path)]
        try:
            relative = member.path.relative_to(input_root)
            candidates.append(normalized_path(relative))
            candidates.append(normalized_path(Path(input_root.name) / relative))
        except ValueError:
            pass
        matched = next((group_mapping[key] for key in candidates if key in group_mapping), None)
        if matched is None:
            missing.append(normalized_path(member.path))
        else:
            group_ids.add(matched)

    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"Patient/study group mapping is missing {len(missing)} images. Examples:\n{preview}")
    if len(group_ids) != 1:
        raise ValueError(
            "Exact duplicate files were assigned different patient/study group IDs: "
            + ", ".join(sorted(group_ids))
        )
    return next(iter(group_ids))


def write_duplicate_audit(
    output_root: Path,
    content_groups: Dict[str, List[SourceImage]],
) -> Tuple[int, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "duplicate_audit.csv"
    duplicate_groups = 0
    conflicting_groups = 0

    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "content_sha256",
                "copy_count",
                "labels",
                "original_splits",
                "has_label_conflict",
                "source_paths_json",
            ]
        )
        for digest, members in sorted(content_groups.items()):
            if len(members) == 1:
                continue
            labels = sorted({member.class_name for member in members})
            splits = sorted({member.original_split for member in members})
            has_conflict = len(labels) > 1
            duplicate_groups += 1
            conflicting_groups += int(has_conflict)
            writer.writerow(
                [
                    digest,
                    len(members),
                    ";".join(labels),
                    ";".join(splits),
                    has_conflict,
                    json.dumps([normalized_path(member.path) for member in members]),
                ]
            )
    return duplicate_groups, conflicting_groups


def prepare_unique_items(
    images: Sequence[SourceImage],
    output_root: Path,
    input_root: Path,
    conflict_policy: str,
    group_mapping: Dict[str, str],
) -> List[Item]:
    by_content: DefaultDict[str, List[SourceImage]] = defaultdict(list)
    for image in images:
        by_content[image.content_sha256].append(image)

    duplicate_groups, conflicting_groups = write_duplicate_audit(output_root, by_content)
    duplicate_copies = sum(len(members) - 1 for members in by_content.values())
    print(
        f"Content audit: {len(images)} files, {len(by_content)} unique images, "
        f"{duplicate_groups} duplicate groups, {duplicate_copies} extra copies, "
        f"{conflicting_groups} conflicting-label groups."
    )
    print(f"Audit written to: {(output_root / 'duplicate_audit.csv').resolve()}")

    if conflicting_groups and conflict_policy == "error":
        raise ValueError(
            f"Found {conflicting_groups} exact-image groups with conflicting labels. "
            "Review duplicate_audit.csv and correct the source labels. If exclusion is "
            "approved and documented, rerun with --conflict_policy exclude."
        )

    items: List[Item] = []
    excluded_files = 0
    for digest, members in sorted(by_content.items()):
        labels = {member.class_name for member in members}
        if len(labels) > 1:
            excluded_files += len(members)
            continue

        representative = sorted(members, key=lambda member: normalized_path(member.path))[0]
        items.append(
            Item(
                src_path=representative.path,
                class_name=representative.class_name,
                label=representative.label,
                original_split=representative.original_split,
                content_sha256=digest,
                duplicate_paths=tuple(sorted(normalized_path(member.path) for member in members)),
                group_id=resolve_group_id(members, group_mapping, input_root),
            )
        )

    if excluded_files:
        print(f"Excluded {excluded_files} files from conflicting-label duplicate groups.")
    if not items:
        raise ValueError("No usable unique images remain after duplicate handling.")
    return items


def safe_unique_name(item: Item) -> str:
    return f"{item.class_name}_{item.content_sha256[:16]}{item.src_path.suffix.lower()}"


def reset_fold_dirs(output_root: Path) -> None:
    """Remove generated fold directories while preserving the audit report."""
    output_root.mkdir(parents=True, exist_ok=True)
    for path in output_root.glob("fold_*"):
        if path.is_dir():
            shutil.rmtree(path)


def ensure_fold_dirs(fold_root: Path) -> None:
    for split_name in ["train", "validation", "test"]:
        for class_name in CLASS_NAMES:
            (fold_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def grouped_stratified_folds(
    items: Sequence[Item],
    n_splits: int,
    random_state: int,
) -> List[List[int]]:
    """Greedily assign whole groups while balancing both class counts."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    grouped: DefaultDict[str, List[int]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[item.group_id].append(index)
    if len(grouped) < n_splits:
        raise ValueError(f"Only {len(grouped)} independent groups are available for {n_splits} folds.")

    total = Counter(item.label for item in items)
    for label in range(len(CLASS_NAMES)):
        groups_with_label = sum(any(items[i].label == label for i in indices) for indices in grouped.values())
        if groups_with_label < n_splits:
            raise ValueError(
                f"Class {CLASS_NAMES[label]!r} appears in only {groups_with_label} independent groups; "
                f"cannot create {n_splits} grouped folds."
            )

    rng = random.Random(random_state)
    group_entries = []
    for group_id, indices in grouped.items():
        counts = Counter(items[i].label for i in indices)
        group_entries.append((group_id, indices, counts, rng.random()))
    group_entries.sort(key=lambda entry: (-len(entry[1]), -max(entry[2].values()), entry[3]))

    fold_indices: List[List[int]] = [[] for _ in range(n_splits)]
    fold_counts = [Counter() for _ in range(n_splits)]
    target = {label: total[label] / n_splits for label in range(len(CLASS_NAMES))}

    def assignment_score(candidate_fold: int, counts: Counter) -> Tuple[float, int, int]:
        score = 0.0
        for label in range(len(CLASS_NAMES)):
            hypothetical = [
                fold_counts[fold][label] + (counts[label] if fold == candidate_fold else 0)
                for fold in range(n_splits)
            ]
            mean_count = sum(hypothetical) / n_splits
            score += sum((count - mean_count) ** 2 for count in hypothetical) / max(target[label], 1.0)
        hypothetical_sizes = [
            len(fold_indices[fold]) + (sum(counts.values()) if fold == candidate_fold else 0)
            for fold in range(n_splits)
        ]
        mean_size = sum(hypothetical_sizes) / n_splits
        score += sum((size - mean_size) ** 2 for size in hypothetical_sizes) / max(mean_size, 1.0)
        return score, len(fold_indices[candidate_fold]), candidate_fold

    for _, indices, counts, _ in group_entries:
        best_fold = min(range(n_splits), key=lambda fold: assignment_score(fold, counts))
        fold_indices[best_fold].extend(indices)
        fold_counts[best_fold].update(counts)

    return [sorted(indices) for indices in fold_indices]


def grouped_train_val_split(
    items: Sequence[Item],
    candidate_indices: Sequence[int],
    val_size: float,
    random_state: int,
) -> Tuple[List[int], List[int]]:
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    subset = [items[index] for index in candidate_indices]
    approximate_folds = max(2, round(1.0 / val_size))
    approximate_folds = min(approximate_folds, len({item.group_id for item in subset}))
    candidate_val_folds = grouped_stratified_folds(subset, approximate_folds, random_state)
    target_size = len(subset) * val_size
    val_relative = min(candidate_val_folds, key=lambda indices: abs(len(indices) - target_size))
    val_set = {candidate_indices[index] for index in val_relative}
    train_indices = sorted(index for index in candidate_indices if index not in val_set)
    val_indices = sorted(val_set)
    return train_indices, val_indices


def assert_no_group_overlap(items: Sequence[Item], split_map: Dict[str, Sequence[int]]) -> None:
    split_groups = {
        split: {items[index].group_id for index in indices}
        for split, indices in split_map.items()
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise AssertionError(f"Group leakage between {left} and {right}: {len(overlap)} groups")


def create_folds(
    items: Sequence[Item],
    output_root: Path,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> None:
    test_folds = grouped_stratified_folds(items, n_splits=n_splits, random_state=random_state)
    all_indices = set(range(len(items)))

    for fold_number, test_indices in enumerate(test_folds, start=1):
        train_val_indices = sorted(all_indices - set(test_indices))
        train_indices, val_indices = grouped_train_val_split(
            items,
            train_val_indices,
            val_size=val_size,
            random_state=random_state + fold_number,
        )
        split_indices = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        }
        assert_no_group_overlap(items, split_indices)

        fold_root = output_root / f"fold_{fold_number}"
        ensure_fold_dirs(fold_root)
        manifest_rows = []

        for split_name, indices in split_indices.items():
            for index in indices:
                item = items[index]
                destination = fold_root / split_name / item.class_name / safe_unique_name(item)
                shutil.copy2(item.src_path, destination)
                manifest_rows.append(
                    {
                        "source_path": normalized_path(item.src_path),
                        "copied_path": normalized_path(destination),
                        "class_name": item.class_name,
                        "fold_split": split_name,
                        "content_sha256": item.content_sha256,
                        "group_id": item.group_id,
                        "duplicate_count": len(item.duplicate_paths),
                        "duplicate_paths_json": json.dumps(item.duplicate_paths),
                    }
                )

        with (fold_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

        counts = {
            split: Counter(items[index].class_name for index in indices)
            for split, indices in split_indices.items()
        }
        print(
            f"[fold_{fold_number}] "
            + ", ".join(
                f"{split}={len(split_indices[split])} {dict(counts[split])}"
                for split in ["train", "validation", "test"]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-resistant stratified TMJ folds.")
    parser.add_argument("--input_root", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, default=Path("data_5_fold"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conflict_policy",
        choices=["error", "exclude"],
        default="error",
        help="How to handle exact duplicate pixels carrying different labels.",
    )
    parser.add_argument(
        "--groups_csv",
        type=Path,
        default=None,
        help="Optional CSV with source_path,group_id for patient/study-level isolation.",
    )
    args = parser.parse_args()

    images = scan_source_images(args.input_root)
    print(f"Found {len(images)} source image files.")
    group_mapping = load_group_mapping(args.groups_csv, args.input_root)
    items = prepare_unique_items(
        images,
        output_root=args.output_root,
        input_root=args.input_root,
        conflict_policy=args.conflict_policy,
        group_mapping=group_mapping,
    )
    print(f"Using {len(items)} unique, non-conflicting image samples.")

    reset_fold_dirs(args.output_root)
    create_folds(
        items=items,
        output_root=args.output_root,
        n_splits=args.n_splits,
        val_size=args.val_size,
        random_state=args.seed,
    )
    print(f"\nDone. Fold dataset written to: {args.output_root.resolve()}")
    if args.groups_csv is None:
        print(
            "WARNING: no patient/study mapping was supplied. Exact-image leakage is prevented, "
            "but patient-level independence cannot be guaranteed."
        )


if __name__ == "__main__":
    main()
