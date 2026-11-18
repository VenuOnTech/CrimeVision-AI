"""Calibrate the evidential classifier on a held-out labelled dataset.

Calibration is empirical, not a formal safety guarantee. The script chooses a
probability threshold and an uncertainty rejection limit subject to a requested
maximum empirical error on accepted calibration samples. A separate untouched
test set must still be evaluated before reporting performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data_utils import (
    label_to_index,
    load_annotation_csv,
    open_rgb_image,
)
from src.evidential_fusion import EvidentialPairHead
from src.inference import EvidentialInferenceEngine


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for class_index in (0, 1):
        mask = y_true == class_index
        if not np.any(mask):
            raise ValueError("Calibration requires both MATCH and CONTRADICTION labels")
        recalls.append(float(np.mean(y_pred[mask] == class_index)))
    return float(np.mean(recalls))


def candidate_thresholds(values: np.ndarray) -> List[float]:
    unique = np.unique(values.astype(float))
    if unique.size == 1:
        return [float(unique[0])]
    mids = ((unique[:-1] + unique[1:]) / 2.0).tolist()
    return sorted(set([float(unique[0]), *mids, float(unique[-1])]))


def choose_selective_thresholds(
    y_true: np.ndarray,
    p_contradiction: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    max_error_rate: float,
    min_accepted_per_class: int,
    quality_in_domain: np.ndarray | None = None,
    max_class_error_rate: float = 0.10,
) -> Tuple[float, float, Dict[str, float | int]]:
    y_true = np.asarray(y_true)
    p_contradiction = np.asarray(p_contradiction, dtype=float)
    epistemic_uncertainty = np.asarray(epistemic_uncertainty, dtype=float)
    if y_true.ndim != 1 or p_contradiction.shape != y_true.shape or epistemic_uncertainty.shape != y_true.shape:
        raise ValueError("labels, probabilities and uncertainty must be one-dimensional arrays of equal length")
    if y_true.size == 0 or set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Calibration requires non-empty data containing both binary classes 0 and 1")
    if not np.isfinite(p_contradiction).all() or not np.isfinite(epistemic_uncertainty).all():
        raise ValueError("Calibration probabilities and uncertainty must be finite")
    if np.any((p_contradiction < 0.0) | (p_contradiction > 1.0)):
        raise ValueError("Contradiction probabilities must be within [0, 1]")
    if np.any((epistemic_uncertainty < 0.0) | (epistemic_uncertainty > 1.0)):
        raise ValueError("Epistemic uncertainty must be within [0, 1]")
    if not np.isfinite(max_error_rate) or not 0.0 <= max_error_rate < 1.0:
        raise ValueError("max_error_rate must be finite and within [0, 1)")
    if not np.isfinite(max_class_error_rate) or not 0.0 <= max_class_error_rate < 1.0:
        raise ValueError("max_class_error_rate must be finite and within [0, 1)")
    if min_accepted_per_class < 1:
        raise ValueError("min_accepted_per_class must be positive")
    best = None
    if quality_in_domain is None:
        quality_in_domain = np.ones(len(y_true), dtype=bool)
    else:
        quality_in_domain = np.asarray(quality_in_domain, dtype=bool)
    if quality_in_domain.shape != y_true.shape:
        raise ValueError("quality_in_domain mask must have one entry per calibration sample")
    for uncertainty_limit in candidate_thresholds(epistemic_uncertainty):
        accepted = (epistemic_uncertainty <= uncertainty_limit) & quality_in_domain
        accepted_labels = y_true[accepted]
        accepted_count = int(accepted.sum())
        if accepted_count == 0:
            continue
        if int(np.sum(accepted_labels == 0)) < min_accepted_per_class:
            continue
        if int(np.sum(accepted_labels == 1)) < min_accepted_per_class:
            continue

        for probability_threshold in candidate_thresholds(p_contradiction[accepted]):
            predictions = (p_contradiction[accepted] >= probability_threshold).astype(int)
            error_rate = float(np.mean(predictions != accepted_labels))
            if error_rate > max_error_rate:
                continue
            accepted_match = accepted_labels == 0
            accepted_contradiction = accepted_labels == 1
            false_positive_rate = float(np.mean(predictions[accepted_match] == 1))
            false_negative_rate = float(np.mean(predictions[accepted_contradiction] == 0))
            if max(false_positive_rate, false_negative_rate) > max_class_error_rate:
                continue
            bal_acc = balanced_accuracy(accepted_labels, predictions)
            coverage = accepted_count / len(y_true)
            candidate = (
                coverage,
                bal_acc,
                -error_rate,
                -abs(probability_threshold - 0.5),
                probability_threshold,
                uncertainty_limit,
                {
                    "accepted_sample_count": accepted_count,
                    "coverage": float(coverage),
                    "empirical_error_rate": error_rate,
                    "false_positive_rate": false_positive_rate,
                    "false_negative_rate": false_negative_rate,
                    "balanced_accuracy": bal_acc,
                    "accepted_match_count": int(np.sum(accepted_labels == 0)),
                    "accepted_contradiction_count": int(np.sum(accepted_labels == 1)),
                },
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "No selective operating point satisfied the overall and per-class empirical error limits "
            "and minimum accepted samples per class. Add more diverse calibration examples or "
            "train a better model. No calibration artifact was written."
        )
    return float(best[4]), float(best[5]), best[6]




def calibrate_binary_baseline(values: np.ndarray, y_true: np.ndarray) -> Tuple[float, str, float]:
    """Choose a one-dimensional baseline threshold/direction on calibration data only."""
    values = np.asarray(values, dtype=float)
    y_true = np.asarray(y_true)
    if values.ndim != 1 or y_true.shape != values.shape or values.size == 0:
        raise ValueError("Baseline values and labels must be non-empty 1-D arrays of equal length")
    if not np.isfinite(values).all() or set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Baseline calibration requires finite values and both binary classes")
    best = None
    for direction in ("below_or_equal", "above_or_equal"):
        for threshold in candidate_thresholds(values):
            if direction == "below_or_equal":
                predictions = (values <= threshold).astype(int)
            else:
                predictions = (values >= threshold).astype(int)
            score = balanced_accuracy(y_true, predictions)
            key = (score, -abs(float(threshold)), direction == "below_or_equal")
            if best is None or key > best[0]:
                best = (key, float(threshold), direction)
    if best is None:
        raise ValueError("Cannot calibrate baseline on an empty value array")
    return best[1], best[2], float(best[0][0])

def calibrate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    calibration_csv = Path(args.calibration_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != args.model_name:
        raise ValueError("--model-name must match the model used to train the checkpoint")
    if checkpoint.get("class_to_index") != {"MATCH": 0, "CONTRADICTION": 1}:
        raise ValueError("Checkpoint has an incompatible class mapping")
    if int(checkpoint.get("feature_schema_version", -1)) != EvidentialInferenceEngine.FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint uses an incompatible feature schema")
    if tuple(checkpoint.get("quality_feature_names", [])) != EvidentialInferenceEngine.QUALITY_FEATURE_NAMES:
        raise ValueError("Checkpoint quality-feature schema is incompatible")
    if float(checkpoint.get("quality_feature_clip_abs_z", -1)) != EvidentialInferenceEngine.QUALITY_FEATURE_CLIP:
        raise ValueError("Checkpoint quality-feature clipping is incompatible")

    rows = load_annotation_csv(calibration_csv)
    training_and_validation_cameras = set(checkpoint.get("train_camera_ids", [])) | set(
        checkpoint.get("validation_camera_ids", [])
    )
    calibration_cameras = {str(row["camera_id"]) for row in rows}
    overlap = training_and_validation_cameras.intersection(calibration_cameras)
    if overlap:
        raise ValueError(
            "Calibration camera IDs overlap the training/validation cameras: "
            f"{sorted(overlap)[:5]}. Use a camera-disjoint calibration set."
        )
    training_and_validation_environments = set(checkpoint.get("train_environment_ids", [])) | set(
        checkpoint.get("validation_environment_ids", [])
    )
    training_and_validation_sources = set(checkpoint.get("train_source_recording_ids", [])) | set(
        checkpoint.get("validation_source_recording_ids", [])
    )
    calibration_environments = {str(row["environment_id"]) for row in rows}
    calibration_sources = {str(row["source_recording_id"]) for row in rows}
    source_overlap = training_and_validation_sources.intersection(calibration_sources)
    if source_overlap:
        raise ValueError(
            "Calibration source_recording_id values overlap the training/validation recordings: "
            f"{sorted(source_overlap)[:5]}. Use source-recording-disjoint data splits."
        )
    environment_overlap = training_and_validation_environments.intersection(calibration_environments)
    if environment_overlap:
        raise ValueError(
            "Calibration environment IDs overlap the training/validation environments: "
            f"{sorted(environment_overlap)[:5]}. Use an environment-disjoint calibration set."
        )

    device = args.device
    resolved_revision = args.model_revision or checkpoint.get("model_revision")
    if not resolved_revision:
        raise ValueError("Checkpoint has no resolved CLIP model revision; retrain with a versioned checkpoint")
    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=resolved_revision,
        device=device,
    )
    if checkpoint.get("model_revision") != engine.model_revision:
        raise ValueError("Checkpoint CLIP revision does not match the currently loaded backbone")
    head = EvidentialPairHead(input_dim=int(checkpoint["input_dim"])).to(engine.device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()

    pair_features = []
    quality_features = []
    global_similarities = []
    standard_ot_costs = []
    labels = []
    for index, row in enumerate(rows, start=1):
        image = open_rgb_image(Path(row["image_path"]))
        extracted = engine.extract_pair_features(image, str(row["statement"]))
        pair_features.append(extracted["pair_features"].float())
        quality_features.append(extracted["quality_features"].float())
        global_similarities.append(float(extracted["global_similarity"]))
        standard_ot_costs.append(float(extracted["standard_entropic_ot_cost"]))
        labels.append(label_to_index(str(row["label"])))
        if index % 10 == 0 or index == len(rows):
            print(f"[calibration] extracted {index}/{len(rows)} examples")

    raw_quality = torch.stack(quality_features)
    quality_mean = torch.as_tensor(checkpoint["quality_feature_mean"], dtype=torch.float32, device=engine.device)
    quality_std = torch.as_tensor(checkpoint["quality_feature_std"], dtype=torch.float32, device=engine.device)
    quality_z = engine.standardize_quality_features(raw_quality.to(engine.device), quality_mean, quality_std)
    x = torch.cat([torch.stack(pair_features).to(engine.device), quality_z], dim=1)
    y_true = np.asarray(labels, dtype=np.int64)
    with torch.inference_mode():
        alpha, probabilities, uncertainty = head(x)
    if (
        not torch.isfinite(alpha).all()
        or not torch.isfinite(probabilities).all()
        or not torch.isfinite(uncertainty).all()
    ):
        raise RuntimeError("Evidential classifier produced non-finite calibration outputs")
    if torch.any(alpha < 1.0) or torch.any(probabilities < 0.0) or torch.any(probabilities > 1.0):
        raise RuntimeError("Evidential classifier produced invalid calibration outputs")
    p_contradiction = probabilities[:, 1].detach().cpu().numpy().astype(float)
    uncertainty_values = uncertainty.detach().cpu().numpy().astype(float)
    similarity_values = np.asarray(global_similarities, dtype=float)
    quality_z = (raw_quality.to(engine.device) - quality_mean) / quality_std
    quality_in_domain = (
        torch.all(torch.abs(quality_z) <= EvidentialInferenceEngine.QUALITY_FEATURE_CLIP, dim=1)
        .detach().cpu().numpy().astype(bool)
    )

    if set(y_true.tolist()) != {0, 1}:
        raise ValueError("Calibration CSV must contain both MATCH and CONTRADICTION examples")

    probability_threshold, uncertainty_limit, selective_metrics = choose_selective_thresholds(
        y_true=y_true,
        p_contradiction=p_contradiction,
        epistemic_uncertainty=uncertainty_values,
        max_error_rate=args.max_error_rate,
        min_accepted_per_class=args.min_accepted_per_class,
        quality_in_domain=quality_in_domain,
        max_class_error_rate=args.max_class_error_rate,
    )

    # Calibrate two simple baselines on the same calibration data only:
    # global CLIP cosine similarity and ordinary (non-debiased) entropic OT.
    # Their thresholds/directions are dataset-specific, not universal.
    global_threshold, global_direction, global_balanced_accuracy = calibrate_binary_baseline(
        similarity_values, y_true
    )
    standard_ot_values = np.asarray(standard_ot_costs, dtype=float)
    standard_ot_threshold, standard_ot_direction, standard_ot_balanced_accuracy = calibrate_binary_baseline(
        standard_ot_values, y_true
    )

    calibration_result = {
        "schema_version": 1,
        "model_name": args.model_name,
        "model_revision": engine.model_revision,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "feature_schema_version": EvidentialInferenceEngine.FEATURE_SCHEMA_VERSION,
        "quality_feature_names": list(EvidentialInferenceEngine.QUALITY_FEATURE_NAMES),
        "quality_feature_clip_abs_z": EvidentialInferenceEngine.QUALITY_FEATURE_CLIP,
        "class_to_index": {"MATCH": 0, "CONTRADICTION": 1},
        "contradiction_probability_threshold": probability_threshold,
        "max_epistemic_uncertainty": uncertainty_limit,
        "max_empirical_error_rate_target": args.max_error_rate,
        "max_empirical_class_error_rate_target": args.max_class_error_rate,
        "calibration_sample_count": len(rows),
        "calibration_quality_in_domain_count": int(np.sum(quality_in_domain)),
        "calibration_quality_out_of_domain_count": int(np.sum(~quality_in_domain)),
        "calibration_class_counts": {
            "MATCH": int(np.sum(y_true == 0)),
            "CONTRADICTION": int(np.sum(y_true == 1)),
        },
        "calibration_camera_ids": sorted(calibration_cameras),
        "calibration_environment_ids": sorted(calibration_environments),
        "calibration_source_recording_ids": sorted(calibration_sources),
        "calibration_csv_sha256": sha256_file(calibration_csv),
        "selective_calibration_metrics": selective_metrics,
        "global_clip_similarity_threshold": global_threshold,
        "global_clip_contradiction_direction": global_direction,
        "global_clip_baseline_balanced_accuracy": global_balanced_accuracy,
        "standard_entropic_ot_threshold": standard_ot_threshold,
        "standard_entropic_ot_contradiction_direction": standard_ot_direction,
        "standard_entropic_ot_baseline_balanced_accuracy": standard_ot_balanced_accuracy,
        "warning": (
            "Empirical calibration only; not a statistical guarantee. Evaluate once on a separate, "
            "untouched camera- and environment-disjoint test set before reporting model performance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration_result, handle, indent=2)
    print(f"Saved calibration artifact: {output_path}")
    print(json.dumps(selective_metrics, indent=2))
    print(f"Contradiction probability threshold: {probability_threshold:.6f}")
    print(f"Maximum accepted epistemic uncertainty: {uncertainty_limit:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--output", default="models/calibration.json")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None, help="Immutable Hugging Face revision/commit; defaults to CLIP_MODEL_REVISION or main")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-error-rate", type=float, default=0.10)
    parser.add_argument("--max-class-error-rate", type=float, default=0.10)
    parser.add_argument("--min-accepted-per-class", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 <= args.max_error_rate < 1.0:
        parser.error("max-error-rate must be in [0, 1)")
    if not 0.0 <= args.max_class_error_rate < 1.0:
        parser.error("max-class-error-rate must be in [0, 1)")
    if args.min_accepted_per_class < 1:
        parser.error("min-accepted-per-class must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


if __name__ == "__main__":
    calibrate(parse_args())
