#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split

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


def create_folds(
    items: List[Item],
    output_root: Path,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> None:
    labels = [item.label for item in items]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(range(len(items)), labels), start=1):
        fold_root = output_root / f"fold_{fold_idx}"
        ensure_fold_dirs(fold_root)

        train_val_items = [items[i] for i in train_val_idx]
        train_val_labels = [item.label for item in train_val_items]

        rel_train_idx, rel_val_idx = train_test_split(
            list(range(len(train_val_items))),
            test_size=val_size,
            stratify=train_val_labels,
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
