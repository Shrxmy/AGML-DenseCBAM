#!/usr/bin/env python
"""Aggregate the four thesis cases and produce Chapter IV statistics/plots.

Expected case directories under --results_root:
    benchmark_clean, benchmark_artifact_mix,
    proposed_clean, proposed_artifact_mix
Each directory must contain all_fold_results.csv produced by the isolated runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CASES: Dict[str, Tuple[str, str]] = {
    "C1 Benchmark Clean": ("benchmark", "clean"),
    "C2 Benchmark Artifact Mix": ("benchmark", "artifact_mix"),
    "C3 Proposed Clean": ("proposed", "clean"),
    "C4 Proposed Artifact Mix": ("proposed", "artifact_mix"),
}
PRIMARY_METRICS = ["accuracy", "precision", "recall", "specificity", "f1"]
EFFICIENCY_METRICS = ["images_per_second", "latency_ms"]
CALIBRATION_METRICS = ["ece_10_bins"]
CORRUPTION_METRICS = [
    "tmd_accuracy_none",
    "tmd_accuracy_motion_blur",
    "tmd_accuracy_gaussian_noise",
    "tmd_accuracy_metal_streak",
]
AUXILIARY_METRICS = [
    "artifact_accuracy",
    "artifact_macro_f1",
    "artifact_recall_none",
    "artifact_recall_motion_blur",
    "artifact_recall_gaussian_noise",
    "artifact_recall_metal_streak",
]


def load_cases(results_root: Path, expected_folds: int) -> pd.DataFrame:
    frames = []
    for case_name, (model_type, scenario) in CASES.items():
        path = results_root / f"{model_type}_{scenario}" / "all_fold_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing case results: {path}")
        frame = pd.read_csv(path)
        required = {
            "fold",
            "model_type",
            "scenario",
            *PRIMARY_METRICS,
            *EFFICIENCY_METRICS,
            *CALIBRATION_METRICS,
            *CORRUPTION_METRICS,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if len(frame) != expected_folds or frame["fold"].nunique() != expected_folds:
            raise ValueError(
                f"{path} must contain exactly {expected_folds} unique folds; "
                f"found {len(frame)} rows and {frame['fold'].nunique()} folds."
            )
        if set(frame["model_type"]) != {model_type} or set(frame["scenario"]) != {scenario}:
            raise ValueError(f"Case labels inside {path} do not match its directory.")
        frame = frame.copy()
        frame["case"] = case_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby("fold")["case"].nunique()
    if not (counts == len(CASES)).all():
        raise ValueError("The same fold identifiers must be present in all four cases.")
    return combined


def descriptive_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS + EFFICIENCY_METRICS + CALIBRATION_METRICS + CORRUPTION_METRICS
    if "artifact_accuracy" in combined and combined["artifact_accuracy"].notna().any():
        metrics.extend(metric for metric in AUXILIARY_METRICS if metric in combined)

    for case_name in CASES:
        case_df = combined[combined["case"] == case_name]
        for metric in metrics:
            values = case_df[metric].dropna().astype(float)
            if values.empty:
                continue
            sem = stats.sem(values) if len(values) > 1 else np.nan
            critical = stats.t.ppf(0.975, len(values) - 1) if len(values) > 1 else np.nan
            margin = critical * sem if len(values) > 1 else np.nan
            rows.append(
                {
                    "case": case_name,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "ci95_low": values.mean() - margin,
                    "ci95_high": values.mean() + margin,
                }
            )
    return pd.DataFrame(rows)


def paired_test(
    differences: np.ndarray,
    alternative: str,
) -> Dict[str, float | str]:
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n < 3:
        raise ValueError("At least three paired folds are required for paired inference.")

    shapiro_stat, shapiro_p = stats.shapiro(differences)
    if np.allclose(differences, 0):
        test_name, statistic, p_value = "all differences zero", 0.0, 1.0
    elif shapiro_p > 0.05:
        result = stats.ttest_1samp(differences, popmean=0.0, alternative=alternative)
        test_name, statistic, p_value = "paired t-test", float(result.statistic), float(result.pvalue)
    else:
        try:
            result = stats.wilcoxon(differences, zero_method="wilcox", alternative=alternative)
            test_name, statistic, p_value = "Wilcoxon signed-rank", float(result.statistic), float(result.pvalue)
        except ValueError:
            test_name, statistic, p_value = "Wilcoxon undefined", np.nan, np.nan

    difference_sd = differences.std(ddof=1)
    cohen_dz = differences.mean() / difference_sd if difference_sd > 0 else np.nan
    return {
        "n": n,
        "mean_difference": float(differences.mean()),
        "difference_std": float(difference_sd),
        "shapiro_w": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "selected_test": test_name,
        "test_statistic": statistic,
        "p_value": p_value,
        "cohen_dz": float(cohen_dz),
    }


def model_comparisons(combined: pd.DataFrame, alternative: str) -> pd.DataFrame:
    rows = []
    for scenario in ["clean", "artifact_mix"]:
        scenario_df = combined[combined["scenario"] == scenario]
        for metric in PRIMARY_METRICS:
            pivot = scenario_df.pivot(index="fold", columns="model_type", values=metric).dropna()
            differences = pivot["proposed"].to_numpy() - pivot["benchmark"].to_numpy()
            rows.append(
                {
                    "comparison": f"Proposed - Benchmark ({scenario})",
                    "metric": metric,
                    "alternative": alternative,
                    **paired_test(differences, alternative),
                }
            )
    return pd.DataFrame(rows)


def robustness_comparisons(combined: pd.DataFrame, alternative: str) -> pd.DataFrame:
    """Positive gain means the proposed model degrades less under artifacts."""
    rows = []
    for metric in PRIMARY_METRICS:
        pivot = combined.pivot(index="fold", columns=["model_type", "scenario"], values=metric).dropna()
        benchmark_drop = pivot[("benchmark", "clean")] - pivot[("benchmark", "artifact_mix")]
        proposed_drop = pivot[("proposed", "clean")] - pivot[("proposed", "artifact_mix")]
        robustness_gain = benchmark_drop.to_numpy() - proposed_drop.to_numpy()
        rows.append(
            {
                "comparison": "Benchmark degradation - Proposed degradation",
                "metric": metric,
                "benchmark_mean_drop": float(benchmark_drop.mean()),
                "proposed_mean_drop": float(proposed_drop.mean()),
                "alternative": alternative,
                **paired_test(robustness_gain, alternative),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    subset = summary[summary["metric"].isin(PRIMARY_METRICS)]
    case_names = list(CASES)
    x = np.arange(len(PRIMARY_METRICS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, case_name in enumerate(case_names):
        case = subset[subset["case"] == case_name].set_index("metric").reindex(PRIMARY_METRICS)
        ax.bar(
            x + (index - 1.5) * width,
            case["mean"],
            width,
            yerr=case["std"],
            capsize=3,
            label=case_name,
        )
    ax.set_xticks(x, [metric.replace("_", " ").title() for metric in PRIMARY_METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Five-fold Performance: Mean ± SD")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "chapter4_metric_bars.png", dpi=300)
    plt.close(fig)


def plot_artifact_breakdown(combined: pd.DataFrame, output_dir: Path) -> None:
    artifact_df = combined[combined["scenario"] == "artifact_mix"]
    labels = ["None/Clean", "Motion Blur", "Gaussian Noise", "Metal Streak"]
    x = np.arange(len(CORRUPTION_METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, model_type in enumerate(["benchmark", "proposed"]):
        frame = artifact_df[artifact_df["model_type"] == model_type]
        means = [frame[metric].mean() for metric in CORRUPTION_METRICS]
        stds = [frame[metric].std(ddof=1) for metric in CORRUPTION_METRICS]
        ax.bar(x + (index - 0.5) * width, means, width, yerr=stds, capsize=3, label=model_type.title())
    ax.set_xticks(x, labels)
    ax.set_ylabel("TMD Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Artifact-Mix Accuracy by Synthetic Corruption: Mean ± SD")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "chapter4_artifact_breakdown.png", dpi=300)
    plt.close(fig)


def plot_fold_lines(combined: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["accuracy", "f1"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for case_name in CASES:
            frame = combined[combined["case"] == case_name].sort_values("fold")
            ax.plot(frame["fold"], frame[metric], marker="o", label=case_name)
        ax.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Fold-wise {metric.upper() if metric == 'f1' else metric.title()}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"chapter4_fold_{metric}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final Chapter IV aggregate tables and paired tests.")
    parser.add_argument("--results_root", type=Path, default=Path("chapter4_results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument(
        "--alternative",
        choices=["greater", "two-sided"],
        default="greater",
        help="greater matches the paper's prespecified one-tailed superiority hypotheses.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_cases(args.results_root, args.expected_folds)
    summary = descriptive_summary(combined)
    comparisons = model_comparisons(combined, args.alternative)
    robustness = robustness_comparisons(combined, args.alternative)

    combined.to_csv(output_dir / "all_cases_all_folds.csv", index=False)
    summary.to_csv(output_dir / "chapter4_summary_mean_sd_ci95.csv", index=False)
    comparisons.to_csv(output_dir / "paired_model_comparisons.csv", index=False)
    robustness.to_csv(output_dir / "paired_robustness_comparisons.csv", index=False)
    plot_metric_bars(summary, output_dir)
    plot_artifact_breakdown(combined, output_dir)
    plot_fold_lines(combined, output_dir)

    metadata = {
        "expected_folds": args.expected_folds,
        "alpha": 0.05,
        "alternative": args.alternative,
        "difference_direction": "proposed minus benchmark",
        "robustness_gain_direction": "benchmark clean-to-artifact drop minus proposed clean-to-artifact drop",
        "warning": "With only five folds, normality tests and inferential p-values have low power; report effect sizes and confidence intervals.",
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Descriptive summary ===")
    print(summary.to_string(index=False))
    print("\n=== Paired model comparisons ===")
    print(comparisons.to_string(index=False))
    print("\n=== Robustness comparisons ===")
    print(robustness.to_string(index=False))
    print(f"\nAnalysis written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
