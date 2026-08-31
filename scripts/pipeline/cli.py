"""Command-line options for a benchmark or proposed-model training case."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train one thesis case across 5 folds.")
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
        help="L2 kernel regularization strength. --weight_decay is retained as a deprecated alias.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--tmd_loss_weight", type=float, default=1.0)
    parser.add_argument("--artifact_loss_weight", type=float, default=0.3)
    parser.add_argument("--fold_limit", type=int, default=None, help="Use 1 for smoke test; omit for all folds.")
    parser.add_argument("--single_fold", type=str, default=None, help="Run only one named fold, e.g. fold_2. Used by isolated runner.")
    parser.add_argument(
        "--skip_integrity_check",
        action="store_true",
        help="Debug only: bypass exact-image leakage validation. Never use for final thesis runs.",
    )
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed_float16 when a GPU is available.",
    )
    parser.add_argument(
        "--class_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply balanced TMD sample weights to the training loss.",
    )
    parser.add_argument(
        "--run_config_sha256",
        type=str,
        default=None,
        help="Canonical isolated-run fingerprint supplied by the runner.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path("results/runs") / f"{args.model_type}_{args.scenario}"
    return RunConfig(
        folds_root=args.folds_root,
        output_dir=output_dir,
        model_type=args.model_type,
        scenario=args.scenario,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        random_state=args.seed,
        freeze_backbone=args.freeze_backbone,
        tmd_loss_weight=args.tmd_loss_weight,
        artifact_loss_weight=args.artifact_loss_weight,
        fold_limit=args.fold_limit,
        single_fold=args.single_fold,
        verify_integrity=not args.skip_integrity_check,
        mixed_precision=args.mixed_precision,
        class_weighting=args.class_weighting,
        run_config_sha256=args.run_config_sha256,
    )
