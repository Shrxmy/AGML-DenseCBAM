"""SHA-256 helpers for datasets, configuration, and training source code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Dict[str, object]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_source_sha256(scripts_dir: Path | None = None) -> str:
    """Fingerprint the training entry point and every pipeline implementation module."""
    scripts_dir = scripts_dir or Path(__file__).resolve().parents[1]
    source_files = [scripts_dir / "train_one_case_5fold.py"]
    source_files.extend(sorted((scripts_dir / "pipeline").glob("*.py")))

    digest = hashlib.sha256()
    for path in source_files:
        relative_name = path.relative_to(scripts_dir).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative_name)
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()
