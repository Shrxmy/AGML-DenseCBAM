#!/usr/bin/env python
"""Train one thesis case across the prepared folds.

The implementation is split into small modules under ``scripts.pipeline`` so each
stage can be read independently. This file remains the stable command-line entry
point and re-exports the established public functions used by utility scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct execution (``python scripts/train_one_case_5fold.py``) needs the
# repository root on sys.path before importing the ``scripts`` package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pipeline.artifacts import (
    add_gaussian_noise,
    add_metal_streak,
    add_metal_streak_v1,
    add_motion_blur,
    apply_artifact,
    densenet_preprocess,
    ensure_uint8,
)
from scripts.pipeline.cli import parse_args
from scripts.pipeline.config import (
    ARTIFACT_LABELS,
    ARTIFACT_PROTOCOL,
    DISPLAY_LABELS,
    IMAGE_EXTENSIONS,
    TMD_LABELS,
    TRAINING_PROTOCOL,
    RunConfig,
    configure_tensorflow,
    effective_run_config_sha256,
    seed_everything,
)
from scripts.pipeline.data import (
    TMJSequence,
    balanced_class_weights,
    index_split_dataset,
    validate_manifest_integrity,
    validate_split_integrity,
)
from scripts.pipeline.evaluation import (
    collect_predictions,
    expected_calibration_error,
    save_learning_curves,
    unpack_batch,
)
from scripts.pipeline.experiment import run_case, run_one_fold
from scripts.pipeline.models import (
    AttentionBlock,
    build_benchmark_model,
    build_proposed_model,
    compile_model,
    fit_model,
    make_backbone,
)
from scripts.pipeline.provenance import sha256_file, sha256_json, training_source_sha256


if __name__ == "__main__":
    run_case(parse_args())
