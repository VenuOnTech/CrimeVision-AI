"""Evaluate the research engine on held-out data.

Modes:
- evidential head: --checkpoint + --calibration + --test-csv
- conformal:       --conformal + --test-csv  [--sweep-alpha 0.05,0.10,0.20,0.30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.data_utils import (
    label_to_index,
    load_annotation_csv,
    open_rgb_image,
    sha256_file,
    validate_research_splits,
)
from src.inference import EvidentialInferenceEngine


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    class_recalls = []
    if tn + fp:
        class_recalls.append(tn / (tn + fp))
    if tp + fn:
        class_recalls.append(tp / (tp + fn))
    balanced = float(np.mean(class_recalls)) if len(class_recalls) == 2 else None
    return {
        "sample_count": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_contradiction": precision,
        "recall_contradiction": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "f1_contradiction": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "class_counts": {"match": tn + fp, "contradiction": tp + fn},
    }


def grouped_metrics(
    rows: List[Dict[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    group_key: str,
) -> Dict[str, Any]:
    results = {}
    group_values = sorted({str(row[group_key]) for row in rows})
    for group in group_values:
        mask = np.asarray([str(row[group_key]) == group for row in rows], dtype=bool)
        results[group] = binary_metrics(labels[mask], predictions[mask])
    return results


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    test_csv = Path(args.test_csv).expanduser().resolve()
    rows = load_annotation_csv(test_csv)

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    import hashlib
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != calibration.get("checkpoint_sha256"):
        raise ValueError("Calibration artifact does not match the supplied checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Keep source recording IDs as the formal split boundary. Camera and
    # environment overlaps are retained as descriptive metadata only: for the
    # current UCA adapter, camera_id is recording-derived and environment_id
    # represents an activity category rather than an independent environment.
    reference_recording_ids = {
        "train": checkpoint.get("train_source_recording_ids", []),
        "validation": checkpoint.get("validation_source_recording_ids", []),
        "calibration": calibration.get("calibration_source_recording_ids", []),
    }
    split_audit = validate_research_splits(
        {"test": rows},
        reference_recording_ids=reference_recording_ids,
    )

    prior_cameras = (
        set(checkpoint.get("train_camera_ids", []))
        | set(checkpoint.get("validation_camera_ids", []))
        | set(calibration.get("calibration_camera_ids", []))
    )
    prior_environments = (
        set(checkpoint.get("train_environment_ids", []))
        | set(checkpoint.get("validation_environment_ids", []))
        | set(calibration.get("calibration_environment_ids", []))
    )
    test_cameras = {str(row["camera_id"]) for row in rows}
    test_environments = {str(row["environment_id"]) for row in rows}
    test_recordings = {str(row["source_recording_id"]) for row in rows}

    # These overlaps are reported, not rejected, because these fields do not
    # establish independent physical-camera or environment generalization.
    camera_metadata_overlap_count = len(prior_cameras & test_cameras)
    environment_category_overlap_count = len(prior_environments & test_environments)

    if {str(row["label"]) for row in rows} != {"MATCH", "CONTRADICTION"}:
        raise ValueError("Test CSV must contain both MATCH and CONTRADICTION labels")

    resolved_revision = args.model_revision or checkpoint.get("model_revision")
    if not resolved_revision:
        raise ValueError("Checkpoint has no resolved CLIP model revision; retrain with a versioned checkpoint")
    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=resolved_revision,
        device=args.device,
        head_checkpoint_path=str(checkpoint_path),
        calibration_path=str(calibration_path),
    )

    labels = []
    global_similarities = []
    standard_ot_costs = []
    p_contradictions = []
    uncertainties = []
    model_predictions = []
    statuses = []

    for index, row in enumerate(rows, start=1):
        image = open_rgb_image(Path(row["image_path"]))
        output = engine.analyze_image(image, str(row["statement"]))
        labels.append(label_to_index(str(row["label"])))
        global_similarities.append(float(output["global_clip_similarity"]))
        standard_ot_costs.append(float(output["standard_entropic_ot_cost"]))
        model_predictions.append(
            1 if output["assessment_status"] == "CONTRADICTION" else
            0 if output["assessment_status"] == "MATCH" else -1
        )
        statuses.append(output["assessment_status"])
        if output["probabilities"] is None:
            raise RuntimeError("Trained classifier was not applied; check checkpoint and calibration paths")
        p_contradictions.append(float(output["probabilities"][1]))
        uncertainties.append(float(output["model_epistemic_uncertainty"]))
        if index % 10 == 0 or index == len(rows):
            print(f"[test] evaluated {index}/{len(rows)} examples")

    y_true = np.asarray(labels, dtype=np.int64)
    y_model = np.asarray(model_predictions, dtype=np.int64)
    similarities = np.asarray(global_similarities, dtype=float)
    standard_ot_values = np.asarray(standard_ot_costs, dtype=float)
    epistemic = np.asarray(uncertainties, dtype=float)

    accepted = y_model >= 0
    model_report = binary_metrics(y_true[accepted], y_model[accepted])
    model_report["coverage"] = float(np.mean(accepted))
    model_report["abstained_count"] = int(np.sum(~accepted))
    model_report["abstention_rate"] = float(np.mean(~accepted))
    model_report["accepted_mean_epistemic_uncertainty"] = (
        float(np.mean(epistemic[accepted])) if np.any(accepted) else None
    )

    baseline_threshold = float(calibration["global_clip_similarity_threshold"])
    baseline_direction = calibration.get("global_clip_contradiction_direction")
    if baseline_direction == "below_or_equal":
        baseline_predictions = (similarities <= baseline_threshold).astype(int)
    elif baseline_direction == "above_or_equal":
        baseline_predictions = (similarities >= baseline_threshold).astype(int)
    else:
        raise ValueError("Calibration artifact has no valid global CLIP baseline direction")
    baseline_report = binary_metrics(y_true, baseline_predictions)

    standard_ot_threshold = float(calibration["standard_entropic_ot_threshold"])
    standard_ot_direction = calibration.get("standard_entropic_ot_contradiction_direction")
    if standard_ot_direction == "below_or_equal":
        standard_ot_predictions = (standard_ot_values <= standard_ot_threshold).astype(int)
    elif standard_ot_direction == "above_or_equal":
        standard_ot_predictions = (standard_ot_values >= standard_ot_threshold).astype(int)
    else:
        raise ValueError("Calibration artifact has no valid standard OT baseline direction")
    standard_ot_report = binary_metrics(y_true, standard_ot_predictions)

    report = {
        "mode": "evidential_head",
        "model_name": args.model_name,
        "model_revision": engine.model_revision,
        "test_csv": str(test_csv),
        "test_sample_count": len(rows),
        "test_camera_count": len(test_cameras),
        "test_environment_count": len(test_environments),
        "test_source_recording_count": len(test_recordings),
        "split_audit": split_audit,
        "camera_metadata_overlap_with_prior_count": camera_metadata_overlap_count,
        "environment_category_overlap_with_prior_count": environment_category_overlap_count,
        "split_boundary_note": (
            "Source recording IDs are the enforced split boundary. Camera and environment "
            "overlap counts are descriptive only; in the current UCA adapter camera IDs are "
            "recording-derived aliases and environment IDs are activity categories."
        ),
        "calibration_artifact_warning": calibration.get("warning"),
        "baseline_global_clip": {
            "similarity_threshold": baseline_threshold,
            "contradiction_direction": baseline_direction,
            **baseline_report,
            "per_camera": grouped_metrics(rows, y_true, baseline_predictions, "camera_id"),
            "per_environment": grouped_metrics(rows, y_true, baseline_predictions, "environment_id"),
        },
        "standard_entropic_ot_baseline": {
            "regularized_ot_threshold": standard_ot_threshold,
            "contradiction_direction": standard_ot_direction,
            **standard_ot_report,
            "per_camera": grouped_metrics(rows, y_true, standard_ot_predictions, "camera_id"),
            "per_environment": grouped_metrics(rows, y_true, standard_ot_predictions, "environment_id"),
        },
        "proposed_evidential_ot": {
            "probability_threshold": calibration["contradiction_probability_threshold"],
            "max_epistemic_uncertainty": calibration["max_epistemic_uncertainty"],
            **model_report,
            "per_camera_accepted_only": grouped_metrics(
                [row for row, keep in zip(rows, accepted) if keep],
                y_true[accepted],
                y_model[accepted],
                "camera_id",
            ),
            "per_environment_accepted_only": grouped_metrics(
                [row for row, keep in zip(rows, accepted) if keep],
                y_true[accepted],
                y_model[accepted],
                "environment_id",
            ),
        },
        "note": (
            "Metrics are measured only on this supplied held-out test CSV. "
            "Per-group balanced accuracy is null when a group contains only one class. "
            "Results do not establish legal or forensic admissibility."
        ),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report: {output_path}")


def _evaluate_conformal_at_alpha(engine, rows, artifact, alpha: float, features_cache=None):
    """Run predict_set at a specific alpha and collect metrics."""
    from src.conformal import (
        coverage_and_abstention,
        extract_conformal_features,
        prediction_set_to_assessment,
        coverage_by_group,
    )
    artifact_at_alpha = artifact.with_alpha(alpha)

    prediction_sets = []
    true_labels = []
    statuses = []
    per_camera: Dict[str, Dict[str, list]] = {}
    per_environment: Dict[str, Dict[str, list]] = {}

    for index, row in enumerate(rows):
        if features_cache is not None and index < len(features_cache):
            features = features_cache[index]
        else:
            image = open_rgb_image(Path(row["image_path"]))
            output = engine.analyze_image(image, str(row["statement"]))
            features = extract_conformal_features({
                "sinkhorn_divergence": output["sinkhorn_divergence"],
                "discarded_mass": output.get("discarded_mass", 0.0),
                "global_clip_similarity": output["global_clip_similarity"],
            })
            if features_cache is not None:
                features_cache.append(features)

        # Get the prediction set for THIS specific alpha
        pred_set = artifact_at_alpha.predict_set(features.reshape(1, -1))[0]
        
        # BUG FIX: Convert the prediction set to a status string directly.
        # Do NOT use output["assessment_status"] because that uses the original artifact alpha.
        status_at_alpha, _ = prediction_set_to_assessment(pred_set)

        prediction_sets.append(pred_set)
        true_labels.append(label_to_index(str(row["label"])))
        statuses.append(status_at_alpha)

        camera = str(row.get("camera_id", "unknown"))
        environment = str(row.get("environment_id", "unknown"))
        per_camera.setdefault(camera, {"sets": [], "labels": []})
        per_environment.setdefault(environment, {"sets": [], "labels": []})
        per_camera[camera]["sets"].append(pred_set)
        per_camera[camera]["labels"].append(label_to_index(str(row["label"])))
        per_environment[environment]["sets"].append(pred_set)
        per_environment[environment]["labels"].append(label_to_index(str(row["label"])))

    overall = coverage_and_abstention(prediction_sets, true_labels, alpha)
    recording_group_coverage = coverage_by_group(
        prediction_sets, true_labels, [str(row["source_recording_id"]) for row in rows]
    )
    per_camera_metrics = {
        camera: coverage_and_abstention(d["sets"], d["labels"], alpha)
        for camera, d in per_camera.items()
    }
    per_environment_metrics = {
        env: coverage_and_abstention(d["sets"], d["labels"], alpha)
        for env, d in per_environment.items()
    }
    status_counts = {"MATCH": 0, "CONTRADICTION": 0, "INDETERMINATE": 0}
    for status in statuses:
        if status in status_counts:
            status_counts[status] += 1

    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q_hat": artifact_at_alpha.q_hat,
        "overall": overall,
        "recording_group_coverage": recording_group_coverage,
        "per_camera": per_camera_metrics,
        "per_environment": per_environment_metrics,
        "status_counts": status_counts,
    }


def evaluate_conformal(args: argparse.Namespace) -> None:
    from src.conformal import ConformalArtifact

    test_csv = Path(args.test_csv).expanduser().resolve()
    conformal_path = Path(args.conformal).expanduser().resolve()
    if not conformal_path.is_file():
        raise FileNotFoundError(f"conformal artifact not found: {conformal_path}")

    rows = load_annotation_csv(test_csv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=device,
        conformal_artifact_path=str(conformal_path),
    )
    if engine.conformal is None:
        raise RuntimeError("conformal artifact was not loaded by the inference engine")
    artifact = engine.conformal
    if artifact.calibration_unit != "source_recording_id_max":
        raise ValueError(
            "This evaluator requires a recording-group calibrated artifact. "
            "Regenerate it with the current calibrate_conformal.py or ablation scripts."
        )

    split_audit = validate_research_splits(
        {"test": rows},
        reference_recording_ids={
            "fit": artifact.fit_group_ids,
            "calibration": artifact.calibration_group_ids,
        },
    )
    test_recording_ids = {str(row["source_recording_id"]) for row in rows}

    alpha_list: List[float] = [artifact.alpha]
    if args.sweep_alpha:
        alpha_list = [float(v.strip()) for v in args.sweep_alpha.split(",") if v.strip()]
        if not alpha_list:
            raise ValueError("--sweep-alpha produced an empty list")
        for a in alpha_list:
            if not 0.0 < a < 1.0:
                raise ValueError(f"--sweep-alpha value out of (0,1): {a}")

    per_alpha = []
    features_cache: List[np.ndarray] = []
    for alpha in alpha_list:
        print(f"[conformal test] alpha={alpha}")
        per_alpha.append(
            _evaluate_conformal_at_alpha(engine, rows, artifact, alpha, features_cache)
        )

    report = {
        "mode": "conformal",
        "schema_version": 3,
        "model_name": args.model_name,
        "model_revision": engine.model_revision,
        "test_csv": str(test_csv),
        "test_csv_sha256": sha256_file(test_csv),
        "conformal_artifact": str(conformal_path),
        "artifact_alpha": artifact.alpha,
        "artifact_q_hat": artifact.q_hat,
        "fit_sample_count": artifact.fit_sample_count,
        "fit_class_counts": artifact.fit_class_counts,
        "calibration_sample_count": artifact.calibration_sample_count,
        "calibration_class_counts": artifact.calibration_class_counts,
        "calibration_unit": artifact.calibration_unit,
        "calibration_group_count": artifact.calibration_group_count,
        "fit_recording_group_count": len(artifact.fit_group_ids),
        "test_recording_group_count": len(test_recording_ids),
        "test_sample_count": len(rows),
        "split_audit": split_audit,
        "test_camera_count": len(per_alpha[0]["per_camera"]),
        "test_environment_count": len(per_alpha[0]["per_environment"]),
        "runs": per_alpha,
        "note": (
            "Calibration uses the maximum true-label nonconformity score per "
            "source_recording_id. The finite-sample guarantee is simultaneous "
            "within a future recording group under exchangeability of recording "
            "groups constructed by the same protocol. It is not class-conditional, "
            "not a correctness guarantee for an individual accepted prediction, "
            "and not an abstention-rate guarantee. Per-example coverage and selective "
            "accuracy are descriptive metrics. The UCA camera_id field is derived "
            "from recording IDs, not independently verified physical camera IDs."
        ),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved conformal evaluation report: {output_path}")

    if len(per_alpha) > 1:
        print()
        print("Sweep summary (coverage / selective_accuracy / abstention_rate / avg_set_size):")
        header = f"{'alpha':>6} | {'target':>7} | {'emp_cov':>8} | {'sel_acc':>8} | {'abstain':>8} | {'avg_set':>8}"
        print(header)
        print("-" * len(header))
        for run in per_alpha:
            o = run["overall"]
            sel = o["selective_accuracy"]
            print(
                f"{run['alpha']:>6.2f} | {run['target_coverage']:>7.2f} | "
                f"{o['empirical_coverage']:>8.3f} | "
                f"{(f'{sel:.3f}' if sel is not None else 'n/a'):>8} | "
                f"{o['abstention_rate']:>8.3f} | "
                f"{o['average_set_size']:>8.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--conformal", default=None)
    parser.add_argument("--sweep-alpha", default=None,
                        help="comma-separated alphas to sweep, e.g. '0.05,0.10,0.20,0.30'")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", default="reports/research_engine_test_metrics.json")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    has_head = bool(args.checkpoint) or bool(args.calibration)
    has_conformal = bool(args.conformal)
    if has_head and has_conformal:
        parser.error("supply either --checkpoint/--calibration OR --conformal, not both")
    if not has_head and not has_conformal:
        parser.error("supply either --checkpoint/--calibration OR --conformal")
    if has_head and not (args.checkpoint and args.calibration):
        parser.error("--checkpoint and --calibration must be supplied together")
    if has_head and args.sweep_alpha:
        parser.error("--sweep-alpha is only supported with --conformal")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "conformal", None):
        evaluate_conformal(args)
    else:
        evaluate(args)