"""Dataset indexing, leakage checks, class weights, and Keras data loading."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence as KerasSequence, to_categorical

from .artifacts import apply_artifact, densenet_preprocess
from .config import ARTIFACT_LABELS, IMAGE_EXTENSIONS, TMD_LABELS
from .provenance import sha256_file


def validate_split_integrity(df: pd.DataFrame) -> None:
    # Reject exact-image leakage and conflicting labels.
    audit = df.copy()
    audit["content_sha256"] = audit["filepath"].map(lambda value: sha256_file(Path(value)))

    conflicts = audit.groupby("content_sha256")["tmd_label"].nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index)
    split_counts = audit.groupby("content_sha256")["split"].nunique()
    leaking_hashes = set(split_counts[split_counts > 1].index)

    if conflicting_hashes or leaking_hashes:
        raise ValueError(
            "Dataset integrity check failed: "
            f"{len(leaking_hashes)} exact-image groups cross train/validation/test and "
            f"{len(conflicting_hashes)} groups carry conflicting labels. "
            "Regenerate folds with scripts/make_5fold_dataset.py after reviewing duplicate_audit.csv."
        )


def validate_manifest_integrity(root: Path, expected_image_count: int) -> None:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Integrity verification requires the generated fold manifest: {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"fold_split", "content_sha256", "group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Incomplete manifest {manifest_path} is missing {sorted(missing)}. "
            "Regenerate folds with the leakage-safe splitter."
        )
    if len(manifest) != expected_image_count:
        raise ValueError(
            f"Manifest/image count mismatch in {root}: {len(manifest)} rows vs "
            f"{expected_image_count} indexed images."
        )
    leaking_groups = manifest.groupby("group_id")["fold_split"].nunique()
    leaking_groups = leaking_groups[leaking_groups > 1]
    if not leaking_groups.empty:
        raise ValueError(
            f"Patient/study group leakage detected in {root}: "
            f"{len(leaking_groups)} groups cross train/validation/test."
        )


def index_split_dataset(root: Path, verify_integrity: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    class_to_idx = {name: idx for idx, name in enumerate(TMD_LABELS)}
    for split in ["train", "validation", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        for class_name in TMD_LABELS:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "filepath": str(path),
                            "split": split,
                            "class_name": class_name,
                            "tmd_label": class_to_idx[class_name],
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {root}")
    if verify_integrity:
        validate_split_integrity(df)
        validate_manifest_integrity(root, expected_image_count=len(df))
    return df


class TMJSequence(KerasSequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int],
        batch_size: int,
        multi_task: bool,
        scenario: str,
        training: bool,
        seed: int,
        tmd_class_weights: Dict[int, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.scenario = scenario
        self.training = training
        self.seed = seed
        self.tmd_class_weights = tmd_class_weights
        self.shuffle_rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))
        self.epoch = -1
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.training:
            self.epoch += 1
            self.shuffle_rng.shuffle(self.indices)

    def _training_rng(self, filepath: str) -> np.random.Generator:
        sample_id = Path(filepath).name
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{sample_id}".encode("utf-8")
        ).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _evaluation_rng(self, filepath: str) -> np.random.Generator:
        # Fold filenames contain stable source hashes.
        sample_id = Path(filepath).name
        digest = hashlib.sha256(f"{self.seed}:{sample_id}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _artifact_for(self, rng: np.random.Generator) -> int:
        if self.scenario == "clean":
            return 0
        return int(rng.integers(0, len(ARTIFACT_LABELS)))

    def evaluation_artifact_label(self, filepath: str) -> int:
        if self.training:
            raise RuntimeError("Deterministic artifact labels are only available for evaluation generators.")
        return self._artifact_for(self._evaluation_rng(filepath))

    def _load_image(self, filepath: str) -> np.ndarray:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def __getitem__(self, index: int):
        batch_df = self.df.iloc[self.indices[index * self.batch_size : (index + 1) * self.batch_size]]
        images: List[np.ndarray] = []
        tmd_labels: List[int] = []
        artifact_labels: List[int] = []
        for row in batch_df.itertuples(index=False):
            image = self._load_image(row.filepath)
            item_rng = self._training_rng(row.filepath) if self.training else self._evaluation_rng(row.filepath)
            if self.training and item_rng.random() < 0.5:
                image = cv2.flip(image, 1)
            artifact_label = self._artifact_for(item_rng)
            image = densenet_preprocess(apply_artifact(image, artifact_label, item_rng))
            images.append(image)
            tmd_labels.append(int(row.tmd_label))
            artifact_labels.append(int(artifact_label))

        x = np.stack(images)
        y_tmd = to_categorical(np.array(tmd_labels), num_classes=len(TMD_LABELS))
        y_artifact = to_categorical(np.array(artifact_labels), num_classes=len(ARTIFACT_LABELS))
        if self.multi_task:
            targets = {"tmd_output": y_tmd, "artifact_output": y_artifact}
            if self.training and self.tmd_class_weights:
                sample_weights = {
                    "tmd_output": np.asarray(
                        [self.tmd_class_weights[label] for label in tmd_labels],
                        dtype=np.float32,
                    ),
                    "artifact_output": np.ones(len(tmd_labels), dtype=np.float32),
                }
                return x, targets, sample_weights
            return x, targets

        if self.training and self.tmd_class_weights:
            sample_weights = np.asarray(
                [self.tmd_class_weights[label] for label in tmd_labels],
                dtype=np.float32,
            )
            return x, y_tmd, sample_weights
        return x, y_tmd


def balanced_class_weights(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts().to_dict()
    total = len(labels)
    if set(counts) != set(range(len(TMD_LABELS))):
        raise ValueError(f"Training split must contain every TMD class; found counts {counts}")
    return {
        label: total / (len(TMD_LABELS) * count)
        for label, count in counts.items()
    }
