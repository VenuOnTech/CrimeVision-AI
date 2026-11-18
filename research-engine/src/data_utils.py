"""Shared validation/loading helpers for labelled research datasets.

Dataset split auditing is deliberately explicit about the grouping unit. The
UCA adapter's `camera_id` is currently derived from the recording filename, and
`environment_id` is the dataset activity category. Neither is an independently
verified physical camera/environment identifier.
"""

from __future__ import annotations

import csv
import hashlib
import json

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from PIL import Image

REQUIRED_COLUMNS = {
    "image_path", "statement", "label", "camera_id", "environment_id", "source_recording_id"
}
VALID_LABELS = {"MATCH", "CONTRADICTION"}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, streaming large inputs."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash a stable, canonical representation of dataset rows."""
    canonical_fields = (
        "image_path", "statement", "label", "camera_id",
        "environment_id", "source_recording_id", "statement_source_recording_id",
        "statement_source_environment_id",
    )
    normalized = []
    for row in rows:
        item = {}
        for key in canonical_fields:
            value = row.get(key)
            item[key] = str(value) if value is not None else None
        normalized.append(item)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_annotation_csv_metadata(csv_path: str | Path) -> List[Dict[str, str | Path]]:
    """Load and validate annotation metadata without requiring image files.

    Use this for split/provenance audits when data paths are mounted on a
    different machine. Inference workflows should use `load_annotation_csv`,
    which additionally verifies every image can be decoded.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")

    rows: List[Dict[str, str | Path]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            raw_image_path = (row.get("image_path") or "").strip()
            statement = (row.get("statement") or "").strip()
            label = (row.get("label") or "").strip().upper()
            camera_id = (row.get("camera_id") or "").strip()
            environment_id = (row.get("environment_id") or "").strip()
            source_recording_id = (row.get("source_recording_id") or "").strip()
            statement_source_recording_id = (row.get("statement_source_recording_id") or "").strip()
            statement_source_environment_id = (row.get("statement_source_environment_id") or "").strip()

            if not raw_image_path or not statement or not camera_id or not environment_id or not source_recording_id:
                raise ValueError(
                    f"{path.name}:{line_number}: image_path, statement, camera_id, environment_id and source_recording_id are required"
                )
            if label not in VALID_LABELS:
                raise ValueError(
                    f"{path.name}:{line_number}: label must be MATCH or CONTRADICTION; got {label!r}"
                )

            image_path = Path(raw_image_path).expanduser()
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()

            parsed = {
                "image_path": image_path,
                "statement": statement,
                "label": label,
                "camera_id": camera_id,
                "environment_id": environment_id,
                "source_recording_id": source_recording_id,
            }
            if statement_source_recording_id:
                parsed["statement_source_recording_id"] = statement_source_recording_id
            if statement_source_environment_id:
                parsed["statement_source_environment_id"] = statement_source_environment_id
            rows.append(parsed)

    if not rows:
        raise ValueError(f"Annotation CSV contains no examples: {path}")
    return rows


def load_annotation_csv(csv_path: str | Path) -> List[Dict[str, str | Path]]:
    """Load annotation rows and verify every referenced image is decodable."""
    path = Path(csv_path).expanduser().resolve()
    rows = load_annotation_csv_metadata(path)
    for line_number, row in enumerate(rows, start=2):
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"{path.name}:{line_number}: evidence image not found: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"{path.name}:{line_number}: unreadable image {image_path}: {exc}") from exc
    return rows


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def label_to_index(label: str) -> int:
    normalized = str(label).strip().upper()
    if normalized == "MATCH":
        return 0
    if normalized == "CONTRADICTION":
        return 1
    raise ValueError(f"unsupported label {label!r}; expected MATCH or CONTRADICTION")


def assert_group_disjoint(
    named_datasets: Dict[str, List[Dict[str, str | Path]]],
    group_key: str,
    group_label: str,
) -> None:
    """Reject overlap of a grouping variable between research data splits."""
    names = list(named_datasets)
    for index, left_name in enumerate(names):
        left_groups = {str(row[group_key]) for row in named_datasets[left_name]}
        for right_name in names[index + 1:]:
            right_groups = {str(row[group_key]) for row in named_datasets[right_name]}
            overlap = left_groups.intersection(right_groups)
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"{group_label} leakage between {left_name} and {right_name}: {sample}. "
                    f"Use {group_label.lower()}-disjoint splits for the claimed generalization result."
                )


def assert_camera_disjoint(named_datasets: Dict[str, List[Dict[str, str | Path]]]) -> None:
    assert_group_disjoint(named_datasets, "camera_id", "Camera")


def assert_environment_disjoint(named_datasets: Dict[str, List[Dict[str, str | Path]]]) -> None:
    assert_group_disjoint(named_datasets, "environment_id", "Environment")


def assert_source_recording_disjoint(named_datasets: Dict[str, List[Dict[str, str | Path]]]) -> None:
    assert_group_disjoint(named_datasets, "source_recording_id", "Source recording")


def validate_research_splits(
    named_datasets: Dict[str, List[Dict[str, Any]]],
    *,
    require_both_classes: bool = True,
    reference_recording_ids: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, Any]:
    """Validate fit/calibration/test boundaries and return a provenance summary.

    `source_recording_id` is the authoritative grouping unit. Camera and
    environment overlap are reported but not treated as physical-camera or
    environment generalisation checks: in the current UCA adapter camera IDs
    are recording-derived aliases and environment IDs are activity classes.
    """
    reference_recording_ids = reference_recording_ids or {}

    if not named_datasets:
        raise ValueError("at least one row-based dataset is required for a split audit")

    if len(named_datasets) + len(reference_recording_ids) < 2:
        raise ValueError(
            "at least two named splits or reference groups are required for a split audit"
        )

    duplicate_names = set(named_datasets).intersection(reference_recording_ids)
    if duplicate_names:
        raise ValueError(
            f"row-based and reference split names overlap: {sorted(duplicate_names)}"
        )

    for split_name, rows in named_datasets.items():
        if not rows:
            raise ValueError(f"{split_name} split is empty")
        labels = {str(row.get("label", "")).strip().upper() for row in rows}
        if not labels.issubset(VALID_LABELS):
            raise ValueError(f"{split_name} contains unsupported labels: {sorted(labels - VALID_LABELS)}")
        if require_both_classes and labels != VALID_LABELS:
            raise ValueError(f"{split_name} must contain both classes; found {sorted(labels)}")
        missing_group = [
            index for index, row in enumerate(rows)
            if not str(row.get("source_recording_id", "")).strip()
        ]
        if missing_group:
            raise ValueError(f"{split_name} has missing source_recording_id values at rows {missing_group[:5]}")

    # Combine available row-based recording IDs with IDs stored in prior artifacts.
    recording_sets = {
        name: {
            str(row["source_recording_id"]).strip()
            for row in rows
        }
        for name, rows in named_datasets.items()
    }

    for split_name, raw_ids in reference_recording_ids.items():
        values = [str(value).strip() if value is not None else "" for value in raw_ids]

        if not values or any(not value for value in values):
            raise ValueError(
                f"reference split {split_name} contains missing recording IDs"
            )

        recording_sets[split_name] = set(values)

    # Reject recording overlap across both row-based splits and artifact references.
    split_names = list(recording_sets)

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            overlap = recording_sets[left_name] & recording_sets[right_name]

            if overlap:
                raise ValueError(
                    "Source recording leakage between "
                    f"{left_name} and {right_name}: {sorted(overlap)[:5]}"
                )

    overlap_report: Dict[str, Dict[str, int]] = {}

    for key in ("source_recording_id", "camera_id", "environment_id"):
        pairwise: Dict[str, int] = {}

        if key == "source_recording_id":
            values_by_split = recording_sets
        else:
            # Historical artifacts contain recording IDs, not complete camera
            # or environment metadata. Only compare fields that are available.
            values_by_split = {
                name: {
                    str(row.get(key, "")).strip()
                    for row in rows
                }
                for name, rows in named_datasets.items()
            }

        names = list(values_by_split)

        for index, left in enumerate(names):
            for right in names[index + 1:]:
                pairwise[f"{left}__{right}"] = len(
                    values_by_split[left] & values_by_split[right]
                )

        overlap_report[key] = pairwise

    camera_is_recording_alias = all(
        str(row.get("camera_id", "")) == f"cam_{row.get('source_recording_id', '')}"
        for rows in named_datasets.values() for row in rows
    )

    # When annotation-source provenance is supplied, verify that negative
    # descriptions do not reference recordings outside their own partition.
    provenance_flags = [
        "statement_source_recording_id" in row
        for rows in named_datasets.values() for row in rows
    ]
    statement_provenance_available = any(provenance_flags)
    if statement_provenance_available and not all(provenance_flags):
        raise ValueError("statement_source_recording_id must be present for every row or none")
    if statement_provenance_available:
        image_sources_by_split = {
            name: {str(row["source_recording_id"]) for row in rows}
            for name, rows in named_datasets.items()
        }
        for split_name, rows in named_datasets.items():
            referenced_sources = set()
            for row in rows:
                image_source = str(row["source_recording_id"])
                statement_source = str(row.get("statement_source_recording_id", "")).strip()
                if not statement_source:
                    raise ValueError(f"{split_name} has a blank statement_source_recording_id")
                referenced_sources.add(statement_source)
                label = str(row["label"]).strip().upper()
                if label == "MATCH" and statement_source != image_source:
                    raise ValueError("MATCH rows must use their own recording's annotation")
                if label == "CONTRADICTION" and statement_source == image_source:
                    raise ValueError("CONTRADICTION proxy rows must use a different recording's annotation")
                statement_environment = row.get("statement_source_environment_id")
                if statement_environment is not None:
                    if label == "MATCH" and str(statement_environment) != str(row.get("environment_id")):
                        raise ValueError("MATCH rows must use their own activity-category annotation")
                    if label == "CONTRADICTION" and str(statement_environment) == str(row.get("environment_id")):
                        raise ValueError(
                            "CONTRADICTION proxy rows must use a different activity category "
                            "when statement_source_environment_id is supplied"
                        )
            outside = referenced_sources - image_sources_by_split[split_name]
            if outside:
                raise ValueError(
                    f"{split_name} statements reference source recordings outside their own split: "
                    f"{sorted(outside)[:5]}"
                )

    # Exact image/statement duplicates are invalid both within and across splits.
    seen_pairs: Dict[tuple[str, str], tuple[str, int]] = {}

    for split_name, rows in named_datasets.items():
        for row_index, row in enumerate(rows):
            image_path = str(row.get("image_path", "")).strip()
            statement = str(row.get("statement", "")).strip()

            if not image_path or not statement:
                raise ValueError(
                    f"{split_name} row {row_index} has missing image_path or statement"
                )

            # Normalize separators/case and insignificant statement whitespace.
            key = (
                image_path.replace("\\", "/").casefold(),
                " ".join(statement.split()),
            )

            previous = seen_pairs.get(key)

            if previous is not None:
                previous_split, previous_index = previous
                raise ValueError(
                    "duplicate image/statement pair appears in "
                    f"{previous_split} row {previous_index} and "
                    f"{split_name} row {row_index}"
                )

            seen_pairs[key] = (split_name, row_index)

    return {
        "split_names": split_names,
        "splits": {
            name: {
                "sample_count": len(rows),
                "class_counts": {
                    "MATCH": sum(str(row["label"]).strip().upper() == "MATCH" for row in rows),
                    "CONTRADICTION": sum(str(row["label"]).strip().upper() == "CONTRADICTION" for row in rows),
                },
                "source_recording_count": len({str(row["source_recording_id"]) for row in rows}),
                "camera_id_count": len({str(row.get("camera_id", "")) for row in rows}),
                "environment_category_count": len({str(row.get("environment_id", "")) for row in rows}),
            }
            for name, rows in named_datasets.items()
        },
        "reference_splits": {
            name: {
                "source_recording_count": len(ids),
            }
            for name, ids in reference_recording_ids.items()
        },
        "pairwise_group_overlap_counts": overlap_report,
        "source_recording_disjoint": True,
        "camera_id_is_recording_alias": camera_is_recording_alias,
        "camera_generalization_claim_supported": False,
        "physical_camera_identity_verified": False,
        "environment_ids_are_activity_categories": True,
        "statement_source_provenance_available": statement_provenance_available,
        "environment_disjointness_required": False,
        "note": (
            "Source-recording IDs are disjoint. In the current UCA adapter, camera_id is derived "
            "from source_recording_id and is not an independently verified physical camera ID; "
            "environment_id represents an activity category and may overlap across splits."
        ),
    }
