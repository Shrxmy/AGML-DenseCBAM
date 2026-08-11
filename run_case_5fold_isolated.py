#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def combine_outputs(output_dir: Path, fold_names: List[str]) -> None:
    result_files = [output_dir / f"{fold_name}_results.csv" for fold_name in fold_names]
    cm_files = [output_dir / f"{fold_name}_confusion_matrix.csv" for fold_name in fold_names]
    missing = [path for path in result_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected fold results: {missing}")
    cm_files = [path for path in cm_files if path.exists()]

    combined = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)
    combined.to_csv(output_dir / "all_fold_results.csv", index=False)

    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "ece_10_bins", "images_per_second", "latency_ms", "epochs_ran"]
    if "artifact_accuracy" in combined and combined["artifact_accuracy"].notna().any():
        metric_cols.append("artifact_accuracy")
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
    parser.add_argument(
        "--l2_strength",
        "--weight_decay",
        dest="l2_strength",
        type=float,
        default=1e-2,
        help="L2 kernel regularization strength.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--tmd_loss_weight", type=float, default=1.0)
    parser.add_argument("--artifact_loss_weight", type=float, default=0.35)
    parser.add_argument("--fold_limit", type=int, default=None, help="Use 1 for smoke test; omit for all folds.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip folds whose fold_N_results.csv already exists.")
    parser.add_argument("--skip_integrity_check", action="store_true", help="Debug only; never use for final thesis runs.")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class_weighting", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("chapter4_results") / f"{args.model_type}_{args.scenario}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = find_folds(args.folds_root, args.fold_limit)
    script_path = Path(__file__).with_name("train_one_case_5fold.py")
    isolated_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    isolated_config["folds"] = [fold_dir.name for fold_dir in fold_dirs]
    isolated_config["python_executable"] = sys.executable
    config_path = output_dir / "isolated_run_config.json"
    existing_results = [
        output_dir / f"{fold_dir.name}_results.csv"
        for fold_dir in fold_dirs
        if (output_dir / f"{fold_dir.name}_results.csv").exists()
    ]
    if args.skip_existing and existing_results:
        if not config_path.exists():
            raise ValueError("Cannot --skip_existing without the original isolated_run_config.json.")
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        comparison_keys = {
            "folds_root",
            "model_type",
            "scenario",
            "image_size",
            "batch_size",
            "epochs",
            "learning_rate",
            "l2_strength",
            "seed",
            "freeze_backbone",
            "tmd_loss_weight",
            "artifact_loss_weight",
            "mixed_precision",
            "class_weighting",
            "skip_integrity_check",
            "folds",
        }
        mismatches = {
            key: (previous_config.get(key), isolated_config.get(key))
            for key in comparison_keys
            if previous_config.get(key) != isolated_config.get(key)
        }
        if mismatches:
            raise ValueError(f"Existing results use a different configuration: {mismatches}")
    config_path.write_text(json.dumps(isolated_config, indent=2), encoding="utf-8")

    for fold_dir in fold_dirs:
        result_path = output_dir / f"{fold_dir.name}_results.csv"
        if args.skip_existing and result_path.exists():
            existing = pd.read_csv(result_path)
            manifest_path = fold_dir / "manifest.csv"
            expected_fingerprint = sha256_file(manifest_path) if manifest_path.exists() else None
            if (
                "fold_manifest_sha256" not in existing
                or len(existing) != 1
                or existing.iloc[0]["fold_manifest_sha256"] != expected_fingerprint
            ):
                raise ValueError(
                    f"Cannot skip stale or unverifiable result {result_path}; rerun the fold."
                )
            print(f"Skipping {fold_dir.name}; matching result and dataset fingerprint found.")
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
            "--l2_strength",
            str(args.l2_strength),
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
        if args.skip_integrity_check:
            cmd.append("--skip_integrity_check")
        if not args.mixed_precision:
            cmd.append("--no-mixed_precision")
        if not args.class_weighting:
            cmd.append("--no-class_weighting")

        print("\n=== Isolated fold process ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    combine_outputs(output_dir, [fold_dir.name for fold_dir in fold_dirs])


if __name__ == "__main__":
    main()
