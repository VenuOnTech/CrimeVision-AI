"""Small shared utilities for auditable, cache-safe research experiments."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data_utils import sha256_file


def hash_image_assets(image_paths) -> str:
    """Hash the ordered image-byte manifest used by feature extraction."""
    digest = hashlib.sha256()
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image asset missing while fingerprinting cache: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_code_hashes() -> dict[str, str]:
    """Hash feature-producing source modules to invalidate caches on code changes."""
    root = Path(__file__).resolve().parent
    modules = (
        "src/inference.py",
        "src/evidential_fusion.py",
        "src/image_quality.py",
    )
    return {name: sha256_file(root / name) for name in modules}


def cache_fingerprint(
    *,
    csv_path: Path,
    model_name: str,
    model_revision: str | None,
    feature_schema_version: int,
    inference_config: Mapping[str, Any],
    selection_config: Mapping[str, Any] | None = None,
    image_paths=None,
) -> str:
    """Fingerprint every known input that can change extracted features."""
    payload = {
        "csv_sha256": sha256_file(csv_path),
        "image_assets_sha256": hash_image_assets(image_paths) if image_paths is not None else None,
        "feature_code_sha256": _feature_code_hashes(),
        "model_name": model_name,
        "model_revision": model_revision,
        "feature_schema_version": int(feature_schema_version),
        "inference_config": dict(inference_config),
        "selection_config": dict(selection_config or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_feature_cache(path: Path, fingerprint: str):
    """Load a cache only when its embedded fingerprint matches exactly."""
    with np.load(path, allow_pickle=False) as data:
        if "fingerprint" not in data.files:
            raise ValueError(f"stale feature cache has no fingerprint: {path}")
        stored = str(data["fingerprint"].item())
        if stored != fingerprint:
            raise ValueError(f"feature cache fingerprint mismatch: {path}")
        return np.asarray(data["X"], dtype=np.float64), np.asarray(data["y"], dtype=np.int64)


def save_feature_cache(path: Path, X: np.ndarray, y: np.ndarray, fingerprint: str) -> None:
    """Atomically write a self-identifying NumPy feature cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, X=X, y=y, fingerprint=np.asarray(fingerprint))
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def build_experiment_manifest(
    *,
    script_name: str,
    model_name: str,
    model_revision: str | None,
    feature_schema_version: int,
    split_csvs: Mapping[str, Path],
    split_rows: Mapping[str, list[dict[str, Any]]],
    split_audit: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    from collections import Counter

    return {
        "manifest_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_script": script_name,
        "model": {"name": model_name, "resolved_revision": model_revision},
        "feature_schema_version": int(feature_schema_version),
        "splits": {
            name: {
                "file_name": path.name,
                "sha256": sha256_file(path),
                "image_assets_sha256": hash_image_assets(
                    row["image_path"] for row in split_rows[name]
                ),
                "sample_count": len(split_rows[name]),
                "class_counts": dict(Counter(str(row["label"]).upper() for row in split_rows[name])),
                "recording_count": len({str(row["source_recording_id"]) for row in split_rows[name]}),
            }
            for name, path in split_csvs.items()
        },
        "split_audit": dict(split_audit),
        "configuration": dict(configuration),
        "interpretation": (
            "This manifest establishes input and configuration provenance. It does not establish "
            "annotation validity, forensic suitability, or generalisation to unseen physical cameras."
        ),
    }
