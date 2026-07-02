#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def find_folds(folds_root: Path, fold_limit: int | None) -> List[Path]:
    fold_dirs = sorted(
        [p for p in folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if fold_limit is not None:
        fold_dirs = fold_dirs[:fold_limit]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold folders found under {folds_root}")
    return fold_dirs


def combine_outputs(output_dir: Path) -> None:
    result_files = sorted(output_dir.glob("fold_*_results.csv"), key=lambda p: int(p.name.split("_")[1]))
    cm_files = sorted(output_dir.glob("fold_*_confusion_matrix.csv"), key=lambda p: int(p.name.split("_")[1]))
    if not result_files:
        raise FileNotFoundError(f"No fold result files found in {output_dir}")

    combined = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)
    combined.to_csv(output_dir / "all_fold_results.csv", index=False)

    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "images_per_second", "latency_ms", "epochs_ran"]
    summary = combined[metric_cols].agg(["mean", "std"]).T.reset_index(names="metric")
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    pooled = None
    for cm_path in cm_files:
        cm = pd.read_csv(cm_path, index_col=0)
        pooled = cm.copy() if pooled is None else pooled.add(cm, fill_value=0)
    if pooled is not None:
        pooled.astype(int).to_csv(output_dir / "pooled_confusion_matrix.csv")

    print("\n=== Combined fold results ===")
    print(combined)
    print("\n=== Mean ± SD summary ===")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one thesis case across folds using one fresh Python process per fold. This avoids GPU memory fragmentation between folds."
    )
    parser.add_argument("--folds_root", type=Path, default=Path("data_5_fold"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_type", choices=["benchmark", "proposed"], default="proposed")
    parser.add_argument("--scenario", choices=["clean", "artifact_mix"], default="artifact_mix")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--tmd_loss_weight", type=float, default=1.0)
    parser.add_argument("--artifact_loss_weight", type=float, default=0.35)
    parser.add_argument("--fold_limit", type=int, default=None, help="Use 1 for smoke test; omit for all folds.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip folds whose fold_N_results.csv already exists.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("chapter4_results") / f"{args.model_type}_{args.scenario}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = find_folds(args.folds_root, args.fold_limit)
    script_path = Path(__file__).with_name("train_one_case_5fold.py")

    for fold_dir in fold_dirs:
        if args.skip_existing and (output_dir / f"{fold_dir.name}_results.csv").exists():
            print(f"Skipping {fold_dir.name}; existing result found.")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--folds_root",
            str(args.folds_root),
            "--output_dir",
            str(output_dir),
            "--model_type",
            args.model_type,
            "--scenario",
            args.scenario,
            "--image_size",
            str(args.image_size),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--learning_rate",
            str(args.learning_rate),
            "--weight_decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
            "--tmd_loss_weight",
            str(args.tmd_loss_weight),
            "--artifact_loss_weight",
            str(args.artifact_loss_weight),
            "--single_fold",
            fold_dir.name,
        ]
        if args.freeze_backbone:
            cmd.append("--freeze_backbone")

        print("\n=== Isolated fold process ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    combine_outputs(output_dir)


if __name__ == "__main__":
    main()
