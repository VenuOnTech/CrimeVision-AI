#!/usr/bin/env python3
"""Generate a tiny synthetic dataset to smoke-test the conformal pipeline.

This script writes 216 decodable PNG images and two CSVs (calibration + test)
with the six columns load_annotation_csv() expects. It is NOT a research
dataset. It exists to prove that:
  - calibrate_conformal.py runs end-to-end,
  - the artifact is written,
  - evaluate_research_engine.py --conformal runs end-to-end,
  - the coverage report has the expected schema.

Numbers from this smoke test are not meaningful. Replace the CSV with real
data (UCF-Crime proxy or hand-labelled CCTV frames) for the actual result.

Each generated example is a coloured geometric shape on a white background,
paired with a statement that either matches the shape (label MATCH) or
describes a different shape (label CONTRADICTION). Every image contributes
exactly one MATCH and one CONTRADICTION example, so class balance is 1:1.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 180, 80),
}
DESCRIPTIONS = {
    ("circle", "red"): "A solid red circle centred on a white background.",
    ("square", "red"): "A solid red square centred on a white background.",
    ("triangle", "red"): "A solid red triangle centred on a white background.",
    ("circle", "blue"): "A solid blue circle centred on a white background.",
    ("square", "blue"): "A solid blue square centred on a white background.",
    ("triangle", "blue"): "A solid blue triangle centred on a white background.",
    ("circle", "green"): "A solid green circle centred on a white background.",
    ("square", "green"): "A solid green square centred on a white background.",
    ("triangle", "green"): "A solid green triangle centred on a white background.",
}
COMBINATIONS = list(DESCRIPTIONS.keys())  # 9 combos
SIZE = 224


def draw(shape: str, color_name: str) -> Image.Image:
    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw_ctx = ImageDraw.Draw(image)
    color = COLORS[color_name]
    if shape == "circle":
        draw_ctx.ellipse([40, 40, 184, 184], fill=color)
    elif shape == "square":
        draw_ctx.rectangle([40, 40, 184, 184], fill=color)
    elif shape == "triangle":
        draw_ctx.polygon([(112, 40), (184, 184), (40, 184)], fill=color)
    else:
        raise ValueError(shape)
    return image


def build_split(split_name: str, output_root: Path, images_per_combo: int, seed: int) -> Path:
    rng = random.Random(seed)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for combo in COMBINATIONS:
        shape, color_name = combo
        for index in range(images_per_combo):
            image = draw(shape, color_name)
            file_name = f"{split_name}_{shape}_{color_name}_{index:03d}.png"
            image_path = image_dir / file_name
            image.save(image_path)
            relative = image_path.relative_to(output_root).as_posix()

            other = rng.choice([c for c in COMBINATIONS if c != combo])
            camera_id = f"cam_{combo[0]}_{combo[1]}"
            environment_id = f"env_{color_name}"
            recording_id = f"{split_name}_{shape}_{color_name}"

            rows.append({
                "image_path": relative,
                "statement": DESCRIPTIONS[combo],
                "label": "MATCH",
                "camera_id": camera_id,
                "environment_id": environment_id,
                "source_recording_id": recording_id,
            })
            rows.append({
                "image_path": relative,
                "statement": DESCRIPTIONS[other],
                "label": "CONTRADICTION",
                "camera_id": camera_id,
                "environment_id": environment_id,
                "source_recording_id": recording_id,
            })

    rng.shuffle(rows)
    csv_path = output_root / f"{split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path", "statement", "label",
                "camera_id", "environment_id", "source_recording_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    output_root = Path(__file__).resolve().parent / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    calib = build_split("calibration", output_root, images_per_combo=12, seed=1)
    test = build_split("test", output_root, images_per_combo=12, seed=2)

    with calib.open() as handle:
        calib_rows = list(csv.DictReader(handle))
    with test.open() as handle:
        test_rows = list(csv.DictReader(handle))

    print(f"Wrote {len(calib_rows)} calibration rows -> {calib}")
    print(f"Wrote {len(test_rows)} test rows          -> {test}")
    print(f"Images under {output_root / 'images'}")
    print()
    print("Class balance (calibration):")
    for label in ("MATCH", "CONTRADICTION"):
        count = sum(1 for row in calib_rows if row["label"] == label)
        print(f"  {label}: {count}")
    print("Class balance (test):")
    for label in ("MATCH", "CONTRADICTION"):
        count = sum(1 for row in test_rows if row["label"] == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()