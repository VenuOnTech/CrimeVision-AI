#!/usr/bin/env python3
"""Adapter: UCA (UCF-Crime Annotation) -> CrimeVision conformal CSVs.

Matches the extracted .png frames in Train/ and Test/ folders with the
UCFCrime.json annotations. Extracts one representative image per video,
creates MATCH and CONTRADICTION pairs, and outputs:
  - fit.csv (from Train/ folder)
  - calibration.csv (50% of Test/ folder, disjoint by recording)
  - test.csv (50% of Test/ folder, disjoint by recording)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

FIELDNAMES = [
    "image_path", "statement", "label",
    "camera_id", "environment_id", "source_recording_id",
    "statement_source_recording_id", "statement_source_environment_id",
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(values) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clean_statement(text: str) -> str:
    """Truncate the long JSON description to a concise witness statement."""
    sentences = text.split(".")
    short_text = ". ".join(sentences[:2]).strip()
    if not short_text.endswith("."):
        short_text += "."
    return short_text[:250]

def build_rows(records, rng):
    """Create paired proxy examples without cross-split statement leakage.

    MATCH uses the same recording's annotation. CONTRADICTION is a coarse
    cross-activity-category mismatch: its annotation comes from a different
    recording in the SAME partition and a different UCF-Crime category. These
    are not human-verified witness-statement contradictions.
    """
    rows = []
    for rec in records:
        image_path_string = str(rec["image_path"].resolve())
        common = {
            "image_path": image_path_string,
            "camera_id": f"cam_{rec['stem']}",
            "environment_id": rec["class"],
            "source_recording_id": rec["stem"],
        }
        rows.append({
            **common,
            "statement": rec["statement"],
            "label": "MATCH",
            "statement_source_recording_id": rec["stem"],
            "statement_source_environment_id": rec["class"],
        })

        candidates = [
            other for other in records
            if other["stem"] != rec["stem"] and other["class"] != rec["class"]
        ]
        if not candidates:
            raise ValueError(
                f"Cannot create a cross-category mismatch for {rec['stem']}: "
                "the current split needs at least one other recording from another category."
            )
        other = rng.choice(candidates)
        rows.append({
            **common,
            "statement": other["statement"],
            "label": "CONTRADICTION",
            "statement_source_recording_id": other["stem"],
            "statement_source_environment_id": other["class"],
        })
    rng.shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", default="datasets/uca")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    json_path = root / "UCFCrime.json"
    
    if not json_path.is_file():
        raise SystemExit(f"Cannot find {json_path}. Did you download it?")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 1. Parse the JSON to get the descriptions
    print("Parsing UCFCrime.json...")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = {}
    for item in data:
        video_path = item.get("video", "")
        desc = item.get("description", "")
        if video_path and desc:
            stem = Path(video_path).stem  # e.g., 'Abuse001_x264'
            descriptions[stem] = clean_statement(desc)

    print(f"Found descriptions for {len(descriptions)} videos.")

    # 2. Scan for images and group by recording
    def get_records(folder_path):
        records = []
        if not folder_path.exists():
            return records
        for class_dir in sorted(folder_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            videos = {}
            for img in sorted(class_dir.glob("*.png")):
                stem = img.stem.rsplit('_', 1)[0]
                if stem not in videos:
                    videos[stem] = img
            for stem, img in videos.items():
                if stem in descriptions:
                    records.append({
                        "image_path": img,
                        "statement": descriptions[stem],
                        "class": class_name,
                        "stem": stem
                    })
        return records

    print("Scanning Train/ folder...")
    train_records = get_records(root / "Train")
    print("Scanning Test/ folder...")
    test_records = get_records(root / "Test")

    print(f"Total valid image-annotation pairs found: {len(train_records) + len(test_records)}")

    train_stems = {record["stem"] for record in train_records}
    test_stems_all = {record["stem"] for record in test_records}
    overlap = train_stems.intersection(test_stems_all)
    if overlap:
        raise ValueError(
            "UCA Train/Test folders contain overlapping recording stems: "
            f"{sorted(overlap)[:10]}"
        )

    # 4. Split Test records into Calibration and Test (Recording-Disjoint)
    test_stems = sorted({r["stem"] for r in test_records})
    rng.shuffle(test_stems)
    split_idx = len(test_stems) // 2
    cal_stems = set(test_stems[:split_idx])
    
    cal_records = [r for r in test_records if r["stem"] in cal_stems]
    final_test_records = [r for r in test_records if r["stem"] not in cal_stems]

    # 5. Generate and write the CSVs
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path.name}: {len(rows)} rows")

    print("Building Fit CSV...")
    fit_rows = build_rows(train_records, rng)
    write_csv(output_dir / "fit.csv", fit_rows)

    print("Building Calibration CSV...")
    cal_rows = build_rows(cal_records, rng)
    write_csv(output_dir / "calibration.csv", cal_rows)

    print("Building Test CSV...")
    test_rows = build_rows(final_test_records, rng)
    write_csv(output_dir / "test.csv", test_rows)

    manifest = {
        "manifest_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "UCF-Crime videos plus UCFCrime.json descriptions",
        "random_seed": args.seed,
        "source_annotation_json_sha256": sha256_file(json_path),
        "split_policy": "Train recordings -> fit; Test recordings split by source recording into calibration/test",
        "pairing_policy": {
            "MATCH": "image paired with its own UCF-Crime annotation description",
            "CONTRADICTION": (
                "image paired with an annotation description from a different recording "
                "and a different activity category within the same partition"
            ),
        },
        "label_semantics": "synthetic cross-category description mismatch proxy; not human-verified witness-statement contradiction",
        "not_claimed": [
            "real witness statements",
            "human-adjudicated image-text contradiction labels",
            "forensic-grade ground truth",
            "physical camera identity or camera-disjoint generalisation",
        ],
        "split_counts": {
            "fit": {"samples": len(fit_rows), "recordings": len(train_records)},
            "calibration": {"samples": len(cal_rows), "recordings": len(cal_records)},
            "test": {"samples": len(test_rows), "recordings": len(final_test_records)},
        },
        "recording_id_set_sha256": {
            "fit": stable_id_hash(record["stem"] for record in train_records),
            "calibration": stable_id_hash(record["stem"] for record in cal_records),
            "test": stable_id_hash(record["stem"] for record in final_test_records),
        },
    }
    manifest_path = output_dir / "uca_pairing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote pairing provenance: {manifest_path}")

    print("\nSuccess! You now have fit.csv, calibration.csv, and test.csv.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())