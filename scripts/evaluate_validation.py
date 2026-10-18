#!/usr/bin/env python
"""Score saved fold checkpoints on the VALIDATION split only, with and without hflip TTA.

Gives the V2 validation baseline for the V3 adoption rule. Never reads test rows
into a generator. Writes one CSV; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from scripts.generate_gradcam import load_checkpoint
from scripts.pipeline.data import TMJSequence, index_split_dataset
from scripts.pipeline.evaluation import collect_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", type=Path, required=True, help="e.g. results/final_v2/proposed_artifact_mix")
    parser.add_argument("--folds_root", type=Path, default=Path("data_5_fold"))
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--recipe", choices=["v2", "v3"], default="v2", help="Recipe the checkpoints were trained with.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}")

    rows = []
    for checkpoint in sorted(args.case_dir.glob("fold_*_best.keras")):
        fold = checkpoint.name.split("_")[0] + "_" + checkpoint.name.split("_")[1]
        model, _ = load_checkpoint(checkpoint)
        multi_task = isinstance(model.output, dict)
        scenario = "clean" if "_clean_" in checkpoint.name else "artifact_mix"
        df = index_split_dataset(args.folds_root / fold)
        validation = df[df["split"] == "validation"].reset_index(drop=True)
        generator = TMJSequence(validation, (args.image_height or args.image_size, args.image_size), 8,
                                multi_task, scenario, False, args.seed, recipe=args.recipe)
        for tta in (False, True):
            y_true, y_pred, *_ = collect_predictions(model, generator, multi_task, tta=tta)
            rows.append({"fold": fold, "tta": tta, "n_validation": len(y_true),
                         "validation_accuracy": accuracy_score(y_true, y_pred),
                         "validation_f1": f1_score(y_true, y_pred, zero_division=0)})
            print(rows[-1])
    if not rows:
        raise FileNotFoundError(f"No fold_*_best.keras checkpoints in {args.case_dir}")
    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.groupby("tta")[["validation_accuracy", "validation_f1"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
