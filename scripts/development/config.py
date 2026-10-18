"""Standard-library-only configuration, planning, and filesystem safety.

Plan-only reads source code for fingerprints and stats fold/manifest paths. It
never reads manifest contents, enumerates images, imports TF, or creates output.
The proposed-only backbone accepts image sides >=32 (optionally non-square); unlike the benchmark
fusion branch it needs no multiple-of-32 pool alignment constraint.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from scripts.pipeline.provenance import sha256_file, sha256_json, training_source_sha256

from . import DEVELOPMENT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLD_NAME = re.compile(r"fold_[1-9][0-9]*\Z")
LABELS = ("normal", "subluxation")
SPLITS = ("train", "validation", "test")
EVALUATION_BOUNDARY = {
    "evaluation_split": "validation",
    "checkpoint_selection_split": "validation",
    "validation_reused_after_checkpoint_selection": True,
    "generalization_estimate": False,
    "test_generator_or_prediction": False,
    "test_access": "Execution integrity audit only: enumerate paths and hash raw bytes; no decoding or inference.",
    "plan_access": "Path metadata and source hashes only; no manifest contents or image reads.",
    "historical_exposure": "This runner does not establish lifetime test non-exposure or patient independence.",
}


@dataclass(frozen=True)
class DevelopmentConfig:
    folds_root: Path
    output_dir: Path
    scenario: str = "artifact_mix"
    attention: str = "cbam"
    auxiliary_head: str = "on"
    artifact_loss_weight: float = 0.3
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    l2_strength: float = 0.01
    seed: int = 42
    single_fold: str | None = None
    folds: tuple[str, ...] | None = None
    image_height: int | None = None
    recipe: str = "v2"

    @property
    def input_hw(self) -> tuple[int, int]:
        return (self.image_height or self.image_size, self.image_size)

    @property
    def multi_task(self) -> bool:
        return self.auxiliary_head == "on"

    @property
    def monitor(self) -> str:
        return "val_primary_ce"

    def payload(self) -> dict:
        result = asdict(self)
        # Omit V3-era fields at their defaults so V2 run fingerprints are unchanged.
        if result["image_height"] is None:
            result.pop("image_height")
        if result["recipe"] == "v2":
            result.pop("recipe")
        result["folds_root"] = str(self.folds_root.resolve())
        result["output_dir"] = str(self.output_dir.resolve())
        result.update(
            version=DEVELOPMENT_VERSION, model_type="proposed_ablation",
            tmd_loss_weight=1.0,
            effective_artifact_loss_weight=self.artifact_loss_weight if self.multi_task else None,
            freeze_backbone=False, class_weighting=True, mixed_precision=True,
            verify_integrity=True, optimizer="Adam", initialization="imagenet",
            monitor=self.monitor,
            checkpoint_objective_note=(
                "All cells minimize the same unweighted per-example validation primary categorical "
                "cross-entropy metric, excluding regularization and auxiliary loss. "
                "Output-specific Keras metric keys are normalized to val_primary_ce."
            ),
        )
        return result


def parse_args(argv: list[str] | None = None) -> tuple[DevelopmentConfig, bool]:
    parser = argparse.ArgumentParser(description="Isolated proposed 2x2 ablation; validation ONLY.")
    parser.add_argument("--folds-root", type=Path, default=Path("data_5_fold"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh child directory under results/development_controlled; existing directories refused.")
    parser.add_argument("--scenario", choices=("clean", "artifact_mix"), default="artifact_mix")
    parser.add_argument("--attention", choices=("cbam", "none"), default="cbam")
    parser.add_argument("--auxiliary-head", choices=("on", "off"), default="on")
    parser.add_argument("--artifact-loss-weight", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=224, help="Proposed-only input width (and height unless --image-height), >=32.")
    parser.add_argument("--image-height", type=int, default=None, help="Input height; defaults to --image-size.")
    parser.add_argument("--recipe", choices=("v2", "v3"), default="v2",
                        help="v3: aspect-preserving input, torch preprocessing, mild augmentation, hflip TTA.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--l2-strength", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--single-fold", help="One fold name, e.g. fold_1.")
    selection.add_argument("--folds", nargs="+", help="Explicit unique fold names; omitted means all folds.")
    parser.add_argument("--plan-only", action="store_true", help="Print deterministic JSON; no TensorFlow or dataset reads/writes.")
    args = vars(parser.parse_args(argv))
    plan_only = args.pop("plan_only")
    if args["folds"] is not None:
        args["folds"] = tuple(args["folds"])
    config = DevelopmentConfig(**args)
    validate_config(config)
    return config, plan_only


def validate_config(config: DevelopmentConfig) -> None:
    if config.scenario not in ("clean", "artifact_mix"):
        raise ValueError("Invalid scenario")
    if config.attention not in ("cbam", "none") or config.auxiliary_head not in ("on", "off"):
        raise ValueError("Only the proposed cbam/none x on/off ablation is supported")
    for name in ("image_size", "batch_size", "epochs", "seed"):
        if type(getattr(config, name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if config.image_size < 32 or config.batch_size < 1 or config.epochs < 1:
        raise ValueError("image_size must be >=32; batch_size and epochs must be positive")
    if config.recipe not in ("v2", "v3"):
        raise ValueError("recipe must be v2 or v3")
    if config.image_height is not None and (type(config.image_height) is not int or config.image_height < 32):
        raise ValueError("image_height must be an integer >=32")
    if config.recipe == "v3" and config.image_height is None:
        raise ValueError("recipe v3 requires an explicit --image-height (aspect-preserving input)")
    if not 0 <= config.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    for name in ("learning_rate", "l2_strength", "artifact_loss_weight"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"{name} must be finite and nonnegative (learning rate strictly positive)")
    if config.single_fold is not None and config.folds is not None:
        raise ValueError("single_fold and folds are mutually exclusive")
    selected = (config.single_fold,) if config.single_fold is not None else config.folds
    if selected is not None:
        if not selected or len(set(selected)) != len(selected) or any(not FOLD_NAME.fullmatch(name) for name in selected):
            raise ValueError("Expected unique fold_N names with N >=1")


def overlaps(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def guard_output(config: DevelopmentConfig, project_root: Path = PROJECT_ROOT) -> Path:
    """Refuse old outputs, symlink destinations, protected trees, and input overlap."""
    root = project_root.resolve()
    allowed = root / "results" / "development_controlled"
    candidate = config.output_dir.absolute()
    # Reject symlinks even when they resolve to another location inside the sandbox.
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symlink output ancestor is forbidden: {ancestor}")
    output = candidate.resolve()
    if allowed not in output.parents:
        raise ValueError(f"Output must be a strict child of {allowed}")
    protected = [root / name for name in (
        "base", "data", "data_5_fold", "chapter4_results", "paper", "output",
        "scripts", "tests", "configs", "notebooks", "results/final_v2", "results/chapter4_results",
    )]
    protected.append(config.folds_root)
    if any(overlaps(output, path) for path in protected):
        raise ValueError("Output overlaps a protected tree or fold input")
    if output.exists():
        raise FileExistsError(f"Output must be absent, including empty directories: {output}")
    return output


def reserve_output(config: DevelopmentConfig, project_root: Path = PROJECT_ROOT) -> Path:
    """Atomic leaf mkdir is the reservation sentinel; never reuse an existing run."""
    output = guard_output(config, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard_output(config, project_root)
    output.mkdir(exist_ok=False)
    with (output / "RUN_RESERVED").open("x", encoding="utf-8") as handle:
        handle.write(DEVELOPMENT_VERSION + "\n")
    return output


def discover_folds(config: DevelopmentConfig) -> list[Path]:
    """Only directory/manifest metadata, not manifest contents or images."""
    validate_config(config)
    root = config.folds_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("folds_root must be a directory")
    names = (config.single_fold,) if config.single_fold is not None else config.folds
    if names is None:
        names = tuple(p.name for p in root.iterdir() if FOLD_NAME.fullmatch(p.name) and p.is_dir())
    folds = []
    for name in sorted(names, key=lambda value: int(value.split("_")[1])):
        fold = root / name
        if fold.is_symlink() or not fold.is_dir():
            raise ValueError(f"Missing or symlink fold directory: {fold}")
        manifest = fold / "manifest.csv"
        if manifest.is_symlink() or not manifest.is_file():
            raise FileNotFoundError(f"Missing/nonregular manifest: {manifest}")
        for split in SPLITS:
            for label in LABELS:
                directory = fold / split / label
                if not directory.is_dir() or directory.is_symlink() or (fold / split).is_symlink():
                    raise ValueError(f"Missing or symlink class directory: {directory}")
        folds.append(fold)
    if not folds:
        raise ValueError(f"No fold_N directories under {root}")
    return folds


def validate_index_rows(rows: Iterable[Mapping], fold_root: Path) -> dict:
    """Require nonempty splits/all classes and prevent escaped image paths.

    Counts are derived independently for each split; no equal-sized-split or
    exact 70/15/15 assumption. Only path strings are inspected here.
    """
    counts = {split: {label: 0 for label in LABELS} for split in SPLITS}
    for row in rows:
        split, label = row["split"], row["class_name"]
        if split not in counts or label not in LABELS or int(row["tmd_label"]) != LABELS.index(label):
            raise ValueError("Invalid indexed split/class/label")
        expected = (fold_root / split / label).resolve()
        if expected not in Path(row["filepath"]).resolve().parents:
            raise ValueError("Indexed image escapes its split/class directory")
        counts[split][label] += 1
    if any(count == 0 for labels in counts.values() for count in labels.values()):
        raise ValueError("Every split must be nonempty and contain both classes")
    return {split: {"count": sum(labels.values()), "class_counts": labels} for split, labels in counts.items()}


def validate_manifest_split_counts(rows: Iterable[Mapping], split_metadata: Mapping) -> dict:
    """A correct total alone is insufficient if manifest split sizes disagree."""
    counts = {split: 0 for split in SPLITS}
    for row in rows:
        split = row.get("fold_split")
        if split not in counts:
            raise ValueError("Manifest contains a missing/unknown fold_split")
        counts[split] += 1
    expected = {split: split_metadata[split]["count"] for split in SPLITS}
    if counts != expected:
        raise ValueError(f"Manifest/indexed split cardinality mismatch: {counts} vs {expected}")
    return counts


def source_provenance(project_root: Path = PROJECT_ROOT) -> dict:
    scripts = project_root / "scripts"
    files = [scripts / "train_one_case_5fold.py", scripts / "run_controlled_development.py"]
    files += sorted((scripts / "pipeline").glob("*.py"))
    files += sorted((scripts / "development").glob("*.py"))
    hashes = {path.relative_to(project_root).as_posix(): sha256_file(path) for path in files}
    return {"training_source_sha256": training_source_sha256(scripts),
            "legacy_pipeline_source_sha256": training_source_sha256(scripts),
            "authoritative_executed_source_field": "development_source_sha256",
            "development_source_sha256": sha256_json(hashes), "source_file_sha256": hashes}


def make_plan(config: DevelopmentConfig, project_root: Path = PROJECT_ROOT) -> dict:
    validate_config(config)
    output = guard_output(config, project_root)
    folds = discover_folds(config)
    payload = config.payload()
    commands = []
    for attention in ("cbam", "none"):
        for auxiliary in ("on", "off"):
            cell_output = output.parent / f"{output.name}_{attention}_aux_{auxiliary}"
            guard_output(replace(config, attention=attention, auxiliary_head=auxiliary,
                                 output_dir=cell_output), project_root)
            command = ["python", "scripts/run_controlled_development.py", "--folds-root", payload["folds_root"],
                       "--output-dir", str(cell_output),
                       "--scenario", config.scenario, "--attention", attention, "--auxiliary-head", auxiliary]
            for flag, value in (("artifact-loss-weight", config.artifact_loss_weight), ("image-size", config.image_size),
                                ("batch-size", config.batch_size), ("epochs", config.epochs),
                                ("learning-rate", config.learning_rate), ("l2-strength", config.l2_strength), ("seed", config.seed)):
                command += [f"--{flag}", str(value)]
            command += ["--recipe", config.recipe]
            if config.image_height is not None:
                command += ["--image-height", str(config.image_height)]
            command += ["--folds", *(fold.name for fold in folds)]
            commands.append(command)
    return {"version": DEVELOPMENT_VERSION, "mode": "plan_only", "config": payload,
            "run_config_sha256": sha256_json(payload), "evaluation_boundary": EVALUATION_BOUNDARY,
            "folds": [{"name": fold.name, "path": str(fold), "manifest_path": str(fold / "manifest.csv"),
                       "manifest_contents_checked": False, "cardinality_checked": False} for fold in folds],
            "sources": source_provenance(project_root), "commands_argv": commands,
            "execution_notice": "Commands are not executed. A real run imports TF, may download ImageNet weights, trains, and predicts validation only."}
