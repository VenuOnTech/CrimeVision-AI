#!/usr/bin/env python3
"""Build a group-disjoint proxy CSV from the Kaggle UCF-Crime video dataset.

The Kaggle dataset `odins0n/ucf-crime-dataset` contains video folders per
anomaly class and DOES NOT contain natural-language sentence annotations.
This script uses class-conditional statement templates as a proxy for witness
statements. It exercises the full train -> calibrate -> evaluate pipeline and
produces a reproducible prototype benchmark, but it measures coarse scene-class
mismatches, not fine-grained witness-statement contradiction. State this
limitation in the report and pair it with the hand-labelled forensic probe.

Class-disjoint splits are used so that environment_id (= class) and
camera_id (= video hash) and source_recording_id are all disjoint across
train/validation/calibration/test, satisfying src/data_utils.py validators.
Contradiction statements are drawn only from classes in the same split, so no
statement template crosses a split boundary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python-headless is required. pip install opencv-python-headless") from exc

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

CLASS_STATEMENTS = {
    "Abuse": ["A person is being abused by another person.", "Someone is abusing another person in the scene."],
    "Arrest": ["A person is being arrested by officers.", "Police officers are arresting a person."],
    "Arson": ["A fire has been started deliberately.", "Someone is setting fire to property."],
    "Assault": ["A person is assaulting another person.", "An assault is taking place."],
    "Burglary": ["Someone is breaking into a building.", "A burglary is in progress."],
    "Explosion": ["An explosion has occurred.", "A blast is visible in the scene."],
    "Fighting": ["Two or more people are fighting.", "A fight is taking place."],
    "Normal": ["People are walking normally with no incident.", "The scene is calm and nothing unusual is happening."],
    "NormalVideos": ["People are walking normally with no incident.", "The scene is calm and nothing unusual is happening."],
    "RoadAccidents": ["A road accident has occurred.", "Vehicles have collided on the road."],
    "Robbery": ["A robbery is taking place.", "Someone is robbing another person."],
    "Shooting": ["A person is shooting a firearm.", "Gunfire is visible in the scene."],
    "Shoplifting": ["A person is shoplifting items.", "Someone is stealing goods from a store."],
    "Stealing": ["A person is stealing property.", "Someone is taking something that does not belong to them."],
    "Vandalism": ["A person is vandalising property.", "Someone is damaging property."],
}


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def find_videos(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXT:
            yield path


def sample_frames(video: Path, count: int, out_dir: Path, class_name: str):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    positions = [int((i + 0.5) / count * (total - 1)) for i in range(count)]
    saved = []
    safe_stem = video.stem.replace(" ", "_")
    for idx, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(pos, total - 1)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out = out_dir / f"{class_name}_{safe_stem}_{idx}.jpg"
        if not cv2.imwrite(str(out), frame):
            continue
        saved.append(out)
    cap.release()
    return saved


def assign_split_classes(classes, rng):
    classes = list(classes)
    rng.shuffle(classes)
    n = len(classes)
    if n < 8:
        raise SystemExit(f"Need at least 8 class folders, found {n}")
    # 2 classes per held-out split, remainder to train. All splits >= 2 classes
    # so contradiction statements can always come from another class in-split.
    test = classes[-2:]
    calibration = classes[-4:-2]
    validation = classes[-6:-4]
    train = classes[:-6]
    return {"train": train, "validation": validation, "calibration": calibration, "test": test}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", default="datasets/ucf_crime_proxy")
    ap.add_argument("--frames-per-video", type=int, default=2)
    ap.add_argument("--max-videos-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)

    by_class = {}
    for video in find_videos(root):
        cls = video.parent.name
        if cls not in CLASS_STATEMENTS:
            continue
        by_class.setdefault(cls, []).append(video)

    if not by_class:
        raise SystemExit("No recognised class folders found under --dataset-root")

    split_classes = assign_split_classes(by_class.keys(), rng)
    fieldnames = ["image_path", "statement", "label", "camera_id", "environment_id", "source_recording_id"]

    for split_name, classes in split_classes.items():
        rows = []
        for cls in classes:
            videos = by_class[cls]
            rng.shuffle(videos)
            videos = videos[: args.max_videos_per_class]
            partners = [c for c in classes if c != cls]
            if not partners:
                continue
            for video in videos:
                frames = sample_frames(video, args.frames_per_video, images, cls)
                if not frames:
                    continue
                rec_id = f"{cls}_{sha12(str(video))}"
                cam_id = f"cam_{sha12(str(video))}"
                own_stmt = rng.choice(CLASS_STATEMENTS[cls])
                other_cls = rng.choice(partners)
                other_stmt = rng.choice(CLASS_STATEMENTS[other_cls])
                for frame in frames:
                    rel = frame.relative_to(out).as_posix()
                    rows.append({"image_path": rel, "statement": own_stmt, "label": "MATCH",
                                 "camera_id": cam_id, "environment_id": cls,
                                 "source_recording_id": rec_id})
                    rows.append({"image_path": rel, "statement": other_stmt, "label": "CONTRADICTION",
                                 "camera_id": cam_id, "environment_id": cls,
                                 "source_recording_id": rec_id})
        if not rows:
            print(f"WARNING: {split_name} produced no rows", file=sys.stderr)
            continue
        path = out / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        labels = sorted({r["label"] for r in rows})
        print(f"{split_name}: {len(rows)} rows, classes={sorted(classes)}, labels={labels} -> {path}")
        if labels != ["CONTRADICTION", "MATCH"]:
            print(f"WARNING: {split_name} does not contain both labels", file=sys.stderr)

    print("\nDone. Statements are class templates, not witness testimony. Report this limitation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())