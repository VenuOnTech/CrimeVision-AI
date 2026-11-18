"""Fit a valid (split-disjoint) conformal artifact.

The command accepts either explicit fit/calibration CSVs or one combined
source CSV that it splits by `source_recording_id`. In both modes, calibration
uses the maximum true-label nonconformity score per recording group.

The test set must be a completely separate CSV passed later to
evaluate_research_engine.py --conformal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.conformal import (
    FEATURE_NAMES,
    extract_conformal_features,
    fit_conformal,
)
from src.data_utils import label_to_index, load_annotation_csv, open_rgb_image, sha256_file, validate_research_splits
from src.inference import EvidentialInferenceEngine


def _extract_features(engine, rows):
    features = []
    labels = []
    for index, row in enumerate(rows, start=1):
        image = open_rgb_image(Path(row["image_path"]))
        output = engine.analyze_image(image, str(row["statement"]))
        features.append(extract_conformal_features(output))
        labels.append(label_to_index(str(row["label"])))
        if index % 500 == 0 or index == len(rows):
            print(f"[conformal] extracted features for {index}/{len(rows)} examples")
    return np.stack(features), np.array(labels, dtype=np.int64)


def _split_by_recording(rows, fit_fraction: float, seed: int):
    recording_ids = sorted({str(row["source_recording_id"]) for row in rows})
    if len(recording_ids) < 4:
        raise ValueError(
            f"Need >= 4 distinct source_recording_id values to split fit/calibration "
            f"cleanly; got {len(recording_ids)}. Add more source videos to your CSV."
        )
    rng = random.Random(seed)
    rng.shuffle(recording_ids)
    n_fit = max(1, min(len(recording_ids) - 1, int(round(len(recording_ids) * fit_fraction))))
    fit_ids = set(recording_ids[:n_fit])
    fit_rows = [row for row in rows if str(row["source_recording_id"]) in fit_ids]
    calibration_rows = [row for row in rows if str(row["source_recording_id"]) not in fit_ids]
    if not fit_rows or not calibration_rows:
        raise ValueError(
            "split produced an empty fit or calibration partition; "
            "adjust --fit-fraction or check that your CSV has enough recordings"
        )
    return fit_rows, calibration_rows


def calibrate(args: argparse.Namespace) -> None:
    output_path = Path(args.output).expanduser().resolve()

    if args.fit_csv and args.calibration_split_csv:
        fit_csv = Path(args.fit_csv).expanduser().resolve()
        calibration_csv = Path(args.calibration_split_csv).expanduser().resolve()
        if not fit_csv.is_file() or not calibration_csv.is_file():
            raise FileNotFoundError("Both --fit-csv and --calibration-split-csv must exist")
        fit_rows = load_annotation_csv(fit_csv)
        calibration_rows = load_annotation_csv(calibration_csv)
        split_audit = validate_research_splits({"fit": fit_rows, "calibration": calibration_rows})
        fit_csv_sha = sha256_file(fit_csv)
        calibration_csv_sha = sha256_file(calibration_csv)
        source_csv_sha = None
        split_mode = "separate_csvs"
    else:
        if args.fit_csv or args.calibration_split_csv:
            raise ValueError("Supply both --fit-csv and --calibration-split-csv")
        if not args.calibration_csv:
            raise ValueError(
                "Supply either --fit-csv + --calibration-split-csv, or --calibration-csv "
                "containing both fit and calibration recordings"
            )
        source_csv = Path(args.calibration_csv).expanduser().resolve()
        if not source_csv.is_file():
            raise FileNotFoundError(f"calibration CSV not found: {source_csv}")
        rows = load_annotation_csv(source_csv)
        fit_rows, calibration_rows = _split_by_recording(rows, args.fit_fraction, args.seed)
        fit_csv_sha = None
        calibration_csv_sha = sha256_file(source_csv)
        source_csv_sha = calibration_csv_sha
        split_audit = validate_research_splits({"fit": fit_rows, "calibration": calibration_rows})
        split_mode = "single_source_csv_split_by_recording"

    for name, split in (("fit", fit_rows), ("calibration", calibration_rows)):
        labels = {str(row["label"]) for row in split}
        if labels != {"MATCH", "CONTRADICTION"}:
            raise ValueError(
                f"{name} split must contain both MATCH and CONTRADICTION labels; "
                f"got {sorted(labels)}"
            )

    print(f"Fit rows: {len(fit_rows)}, Calibration rows: {len(calibration_rows)}")
    print(f"Recording groups: fit={split_audit['splits']['fit']['source_recording_count']}, "
          f"calibration={split_audit['splits']['calibration']['source_recording_count']}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.trust_mode == "absolute":
        if args.trust_tau is None or args.trust_k is None or args.trust_k <= 0:
            raise ValueError("--trust-mode absolute requires finite --trust-tau and positive --trust-k")
        import os
        os.environ["OT_TRUST_TAU"] = repr(float(args.trust_tau))
        os.environ["OT_TRUST_K"] = repr(float(args.trust_k))
    elif args.trust_tau is not None or args.trust_k is not None:
        raise ValueError("--trust-tau/--trust-k are only valid with --trust-mode absolute")

    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=device,
        trust_mode=args.trust_mode,
    )

    fit_features, fit_labels = _extract_features(engine, fit_rows)
    calibration_features, calibration_labels = _extract_features(engine, calibration_rows)

    artifact = fit_conformal(
        fit_features=fit_features,
        fit_labels=fit_labels,
        calibration_features=calibration_features,
        calibration_labels=calibration_labels,
        alpha=args.alpha,
        inference_config=engine.feature_config,
        fit_csv_sha256=fit_csv_sha,
        calibration_csv_sha256=calibration_csv_sha,
        cli_model_revision=engine.model_revision,
        feature_schema_version=EvidentialInferenceEngine.FEATURE_SCHEMA_VERSION,
        fit_group_ids=[str(row["source_recording_id"]) for row in fit_rows],
        calibration_group_ids=[str(row["source_recording_id"]) for row in calibration_rows],
    )
    artifact.save(output_path)

    report = {
        "artifact": str(output_path),
        "schema_version": 3,
        "alpha": args.alpha,
        "coverage_guarantee": 1.0 - args.alpha,
        "q_hat": artifact.q_hat,
        "fit_sample_count": artifact.fit_sample_count,
        "fit_class_counts": artifact.fit_class_counts,
        "calibration_sample_count": artifact.calibration_sample_count,
        "calibration_class_counts": artifact.calibration_class_counts,
        "calibration_unit": artifact.calibration_unit,
        "calibration_group_count": artifact.calibration_group_count,
        "fit_csv_sha256": fit_csv_sha,
        "calibration_csv_sha256": calibration_csv_sha,
        "source_csv_sha256": source_csv_sha,
        "split_mode": split_mode,
        "fit_fraction": args.fit_fraction if split_mode != "separate_csvs" else None,
        "split_seed": args.seed if split_mode != "separate_csvs" else None,
        "split_audit": split_audit,
        "feature_names": list(FEATURE_NAMES),
    }
    print(json.dumps(report, indent=2))
    print(
        "\nConformal artifact saved (schema 3: source-recording group calibration). "
        "The reported guarantee is simultaneous within a future recording group under "
        "exchangeability of recording groups; it is not class-conditional, not a "
        "per-case correctness guarantee, and not an abstention-rate guarantee."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-csv", default=None,
                        help="one CSV containing both fit and calibration recordings; split by source recording")
    parser.add_argument("--fit-csv", default=None,
                        help="explicit fit-split CSV; use with --calibration-split-csv")
    parser.add_argument("--calibration-split-csv", default=None,
                        help="explicit calibration-split CSV; use with --fit-csv")
    parser.add_argument("--output", default="models/conformal.json")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--fit-fraction", type=float, default=0.5,
                        help="fraction of source recordings used to fit densities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--trust-mode", choices=("heterogeneity", "absolute"), default="heterogeneity")
    parser.add_argument("--trust-tau", type=float, default=None,
                        help="required with --trust-mode absolute; fit on fit split before freezing")
    parser.add_argument("--trust-k", type=float, default=None,
                        help="required with --trust-mode absolute; fit on fit split before freezing")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1")
    if not 0.1 <= args.fit_fraction <= 0.9:
        parser.error("--fit-fraction must be in [0.1, 0.9]")
    return args


if __name__ == "__main__":
    calibrate(parse_args())