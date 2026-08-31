#!/usr/bin/env python
# Generate learning curves from fold history files.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .train_one_case_5fold import save_learning_curves
except ImportError:  # Direct execution
    from train_one_case_5fold import save_learning_curves


def infer_model_type(case_dir: Path) -> str:
    if case_dir.name.startswith("benchmark_"):
        return "benchmark"
    if case_dir.name.startswith("proposed_"):
        return "proposed"
    raise ValueError(f"Cannot infer model type from case directory: {case_dir}")


def generate_for_results_root(
    results_root: Path,
    output_root: Path | None = None,
) -> list[Path]:
    history_paths = sorted(results_root.glob("*/fold_*_history.csv"))
    if not history_paths:
        raise FileNotFoundError(f"No fold history CSV files found under {results_root}")

    outputs: list[Path] = []
    for history_path in history_paths:
        history = pd.read_csv(history_path)
        numeric = history.select_dtypes(include=np.number)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"History contains NaN/Inf values: {history_path}")
        model_type = infer_model_type(history_path.parent)
        fold_name = history_path.name.removesuffix("_history.csv")
        scenario = history_path.parent.name.removeprefix(f"{model_type}_")
        output_path = (
            output_root / history_path.parent.name / f"{fold_name}_learning_curves.png"
            if output_root is not None
            else history_path.with_name(f"{fold_name}_learning_curves.png")
        )
        save_learning_curves(
            history,
            model_type,
            output_path,
            title=f"{fold_name.replace('_', ' ').title()} — {model_type.title()} / {scenario}",
        )
        outputs.append(output_path)
        print(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fold-level training/validation loss figures from saved histories."
    )
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional separate root that preserves source histories and checksums.",
    )
    args = parser.parse_args()
    outputs = generate_for_results_root(args.results_root, args.output_root)
    print(f"Generated {len(outputs)} learning-curve figures.")


if __name__ == "__main__":
    main()
