#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["normal", "subluxation"]
SOURCE_SPLITS = ["train", "validation", "test"]


@dataclass
class Item:
    src_path: Path
    class_name: str
    label: int
    original_split: str


def scan_dataset(input_root: Path) -> List[Item]:
    items: List[Item] = []
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
                    items.append(
                        Item(
                            src_path=file_path,
                            class_name=class_name,
                            label=class_to_idx[class_name],
                            original_split=split_name,
                        )
                    )

    if not items:
        raise ValueError(f"No images found under {input_root}")
    return items


def safe_unique_name(item: Item) -> str:
    digest = hashlib.md5(str(item.src_path).encode("utf-8")).hexdigest()[:10]
    return f"{item.original_split}_{item.class_name}_{item.src_path.stem}_{digest}{item.src_path.suffix.lower()}"


def reset_output_dir(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def ensure_fold_dirs(fold_root: Path) -> None:
    for split_name in ["train", "validation", "test"]:
        for class_name in CLASS_NAMES:
            (fold_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def copy_item(item: Item, dst_path: Path) -> None:
    shutil.copy2(item.src_path, dst_path)

def stratified_kfold_indices(
    items: List[Item],
    n_splits: int,
    random_state: int,
) -> List[Tuple[List[int], List[int]]]:
    """Return deterministic stratified train/test index pairs without sklearn."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    by_label: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_label[item.label].append(idx)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    rng = random.Random(random_state)
    for label, indices in sorted(by_label.items()):
        if len(indices) < n_splits:
            raise ValueError(
                f"Class label {label} has only {len(indices)} samples; "
                f"cannot create {n_splits} stratified folds."
            )
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for pos, idx in enumerate(shuffled):
            folds[pos % n_splits].append(idx)

    all_indices = set(range(len(items)))
    pairs: List[Tuple[List[int], List[int]]] = []
    for test_indices in folds:
        test_set = set(test_indices)
        train_val_indices = sorted(all_indices - test_set)
        pairs.append((train_val_indices, sorted(test_indices)))
    return pairs


def stratified_train_val_split(
    items: List[Item],
    val_size: float,
    random_state: int,
) -> Tuple[List[int], List[int]]:
    """Split relative indices into train/validation while preserving class ratios."""
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    by_label: DefaultDict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_label[item.label].append(idx)

    rng = random.Random(random_state)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for label, indices in sorted(by_label.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_size))
        if val_count >= len(shuffled):
            raise ValueError(
                f"Validation split for label {label} would consume the whole class."
            )
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return sorted(train_indices), sorted(val_indices)


def create_folds(
    items: List[Item],
    output_root: Path,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> None:
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        stratified_kfold_indices(items, n_splits=n_splits, random_state=random_state),
        start=1,
    ):
        fold_root = output_root / f"fold_{fold_idx}"
        ensure_fold_dirs(fold_root)

        train_val_items = [items[i] for i in train_val_idx]

        rel_train_idx, rel_val_idx = stratified_train_val_split(
            train_val_items,
            val_size=val_size,
            random_state=random_state + fold_idx,
        )

        split_map: Dict[str, List[Item]] = {
            "train": [train_val_items[i] for i in rel_train_idx],
            "validation": [train_val_items[i] for i in rel_val_idx],
            "test": [items[i] for i in test_idx],
        }

        manifest_rows: List[Tuple[str, str, str, str]] = []

        for new_split, split_items in split_map.items():
            for item in split_items:
                filename = safe_unique_name(item)
                dst_path = fold_root / new_split / item.class_name / filename
                copy_item(item, dst_path)
                manifest_rows.append(
                    (
                        str(item.src_path),
                        str(dst_path),
                        item.class_name,
                        new_split,
                    )
                )

        with (fold_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_path", "copied_path", "class_name", "fold_split"])
            writer.writerows(manifest_rows)

        train_n = len(split_map["train"])
        val_n = len(split_map["validation"])
        test_n = len(split_map["test"])
        print(f"[fold_{fold_idx}] train={train_n}, validation={val_n}, test={test_n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create true stratified 5-fold TMJ dataset folders.")
    parser.add_argument("--input_root", type=Path, default=Path("data"), help="Path to original data folder.")
    parser.add_argument("--output_root", type=Path, default=Path("data_5_fold"), help="Path to new 5-fold folder.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction taken from train+val pool inside each fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    items = scan_dataset(args.input_root)
    print(f"Found {len(items)} total images.")
    reset_output_dir(args.output_root)
    create_folds(
        items=items,
        output_root=args.output_root,
        n_splits=args.n_splits,
        val_size=args.val_size,
        random_state=args.seed,
    )
    print(f"\nDone. New fold dataset written to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
