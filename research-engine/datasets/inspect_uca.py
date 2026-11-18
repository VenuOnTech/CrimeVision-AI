#!/usr/bin/env python3
r"""Inspect a downloaded UCA (UCF-Crime Annotation) dataset and print its layout.

This script does NOT modify anything. It exists so that the UCA adapter can be
built against the ACTUAL schema of your specific download, rather than guessed.
Different UCA mirrors distribute slightly different JSON layouts.

Usage:
    py datasets/inspect_uca.py "C:\Capstone-Project\datasets\UCA"

Output to paste back:
    - Top-level directory structure
    - Every JSON file with size and top-level type
    - Full key structure of the first entry in each JSON file, with sample values
    - Video file count, parent-folder names, and one sample filename per folder
    - A filename-match check: does each annotation stem resolve to a video file?

Discipline reminder (this script just reports; it does not enforce):
    Every source_recording_id must end up in exactly one of {fit, calibration,
    test}. The adapter will enforce this. It is stated here so the invariant is
    visible at inspection time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SAMPLE_PREVIEW = 200


def _preview(value: Any) -> str:
    text = repr(value)
    if len(text) > MAX_SAMPLE_PREVIEW:
        return text[:MAX_SAMPLE_PREVIEW] + "..."
    return text


def _describe_structure(value: Any, indent: int = 4, max_depth: int = 2) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{prefix}dict with {len(keys)} keys")
        for key in keys[:10]:
            child = value[key]
            child_type = type(child).__name__
            if isinstance(child, (list, dict)):
                if isinstance(child, list):
                    print(f"{prefix}  '{key}': list[{len(child)}]")
                    if child and max_depth > 0:
                        _describe_structure(child[0], indent + 6, max_depth - 1)
                else:
                    print(f"{prefix}  '{key}': dict[{len(child)}]")
                    if max_depth > 0:
                        _describe_structure(child, indent + 6, max_depth - 1)
            else:
                print(f"{prefix}  '{key}': {child_type} = {_preview(child)}")
        if len(keys) > 10:
            print(f"{prefix}  ... and {len(keys) - 10} more keys")
    elif isinstance(value, list):
        print(f"{prefix}list with {len(value)} entries")
        if value and max_depth > 0:
            print(f"{prefix}first entry:")
            _describe_structure(value[0], indent + 4, max_depth - 1)
    else:
        print(f"{prefix}{type(value).__name__} = {_preview(value)}")


def _count_annotation_entries(data: Any) -> int:
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def _sample_first_stem(data: Any) -> str | None:
    if isinstance(data, dict):
        for key in data:
            return str(key)
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            for key in ("video", "video_name", "filename", "name"):
                if key in item:
                    return str(item[key])
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py datasets/inspect_uca.py <dataset-root>")
        return 2

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 2

    print("=" * 72)
    print(f"UCA DATASET INSPECTION")
    print(f"Root: {root}")
    print("=" * 72)

    # -------- Top-level structure --------
    print("\n[1] TOP-LEVEL DIRECTORY STRUCTURE")
    print("-" * 72)
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            n_subdirs = sum(1 for x in entry.iterdir() if x.is_dir())
            n_files = sum(1 for x in entry.iterdir() if x.is_file())
            print(f"  DIR   {entry.name}/    ({n_subdirs} subdirs, {n_files} files)")
        else:
            size_kb = entry.stat().st_size / 1024
            print(f"  FILE  {entry.name}  ({size_kb:,.1f} KB)")

    # -------- JSON files --------
    json_files = sorted(root.rglob("*.json"))
    print(f"\n[2] ANNOTATION JSON FILES  (found: {len(json_files)})")
    print("-" * 72)

    annotation_sample_stem: str | None = None
    annotation_sample_data: Any = None
    total_entries = 0

    for json_path in json_files:
        rel = json_path.relative_to(root)
        size_mb = json_path.stat().st_size / (1024 * 1024)
        print(f"\n  FILE: {rel}  ({size_mb:,.2f} MB)")
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"    Failed to load: {exc}")
            continue

        n_entries = _count_annotation_entries(data)
        total_entries += n_entries
        print(f"    Top-level type: {type(data).__name__}")
        print(f"    Entry count:    {n_entries}")
        print(f"    First-entry structure:")
        _describe_structure(data, indent=6, max_depth=2)

        if annotation_sample_stem is None:
            annotation_sample_stem = _sample_first_stem(data)
            annotation_sample_data = data

    print(f"\n  Total annotated entries across all JSON files: {total_entries}")

    # -------- Video files --------
    video_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    print(f"\n[3] VIDEO FILES  (found: {len(video_files)})")
    print("-" * 72)

    if not video_files:
        print("  No video files found under the dataset root.")
        print("  Verify extraction completed and the path is correct.")
    else:
        parents: dict[str, list[Path]] = {}
        for path in video_files:
            parent_name = path.parent.name
            parents.setdefault(parent_name, []).append(path)
        print(f"  Distinct parent folders: {len(parents)}")
        for parent_name in sorted(parents):
            sample = parents[parent_name][0]
            rel_sample = sample.relative_to(root)
            print(f"    {parent_name:30s}  {len(parents[parent_name]):4d} files   e.g. {rel_sample.name}")

    # -------- Filename matching --------
    print(f"\n[4] FILENAME MATCH CHECK")
    print("-" * 72)
    if annotation_sample_stem is None:
        print("  Could not obtain a sample annotation stem; skipping.")
    elif not video_files:
        print("  No video files present; cannot verify matching.")
    else:
        video_stems = {p.stem: p for p in video_files}
        sample_stem = annotation_sample_stem
        print(f"  First annotation stem from JSON: {sample_stem!r}")

        # Try exact, then with common suffixes, then prefix match.
        matches = []
        if sample_stem in video_stems:
            matches.append(("exact", video_stems[sample_stem]))
        else:
            for suffix in ("_x264", "-x264", "_x264_", ".x264"):
                candidate = sample_stem + suffix
                if candidate in video_stems:
                    matches.append((f"suffix {suffix!r}", video_stems[candidate]))
                    break
            if not matches:
                for stem, path in video_stems.items():
                    if stem.startswith(sample_stem) or sample_stem.startswith(stem):
                        matches.append(("prefix", path))
                        break

        if matches:
            for how, path in matches:
                print(f"    MATCH ({how}): {path.relative_to(root)}")
        else:
            print("    NO MATCH found.")
            print(f"    Sample video stems for comparison (first 5):")
            for stem in list(video_stems.keys())[:5]:
                print(f"      {stem!r}")

        # Broader match rate.
        matched = 0
        unmatched_sample = []
        for stem in list(video_stems.keys()):
            matched += 1  # placeholder; we cannot cheaply check every annotation
        # Better: count how many annotation stems resolve.
        # For a quick estimate, check a small sample against video_stems.
        if isinstance(annotation_sample_data, dict):
            annotation_stems = list(annotation_sample_data.keys())[:200]
        elif isinstance(annotation_sample_data, list) and annotation_sample_data:
            annotation_stems = []
            for item in annotation_sample_data[:200]:
                if isinstance(item, dict):
                    for key in ("video", "video_name", "filename", "name"):
                        if key in item:
                            annotation_stems.append(str(item[key]))
                            break
        else:
            annotation_stems = []

        if annotation_stems:
            resolved = 0
            for stem in annotation_stems:
                if stem in video_stems:
                    resolved += 1
                    continue
                for suffix in ("_x264", "-x264", "_x264_", ".x264"):
                    if stem + suffix in video_stems:
                        resolved += 1
                        break
                else:
                    if any(
                        vstem.startswith(stem) or stem.startswith(vstem)
                        for vstem in video_stems
                    ):
                        resolved += 1
            print(f"  Match rate over first {len(annotation_stems)} annotation entries: "
                  f"{resolved}/{len(annotation_stems)} resolved to a video file.")

    # -------- Sample annotation entries --------
    print(f"\n[5] SAMPLE ANNOTATION ENTRIES  (first 3)")
    print("-" * 72)
    if annotation_sample_data is None:
        print("  No annotation data available.")
    elif isinstance(annotation_sample_data, dict):
        for i, (key, value) in enumerate(annotation_sample_data.items()):
            if i >= 3:
                break
            print(f"  --- entry {i+1} ---")
            print(f"  key: {key!r}")
            _describe_structure(value, indent=4, max_depth=3)
    elif isinstance(annotation_sample_data, list):
        for i, item in enumerate(annotation_sample_data[:3]):
            print(f"  --- entry {i+1} ---")
            _describe_structure(item, indent=4, max_depth=3)

    print("\n" + "=" * 72)
    print("INSPECTION COMPLETE")
    print("=" * 72)
    print(
        "\nPaste this entire output back into the conversation. The adapter\n"
        "will then be built against exactly the schema your download contains.\n"
        "Do not run any adapter before this output is reviewed.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())