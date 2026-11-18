"""Evaluate fixed-threshold robustness to image distortions.

Fits each conformal artifact once on clean fit/calibration features, then applies
that same artifact and q_hat to clean and distorted test images. No corrupted
images are used for fitting or calibration.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.conformal import (
    FEATURE_NAMES,
    coverage_and_abstention,
    coverage_by_group,
    fit_conformal,
)
from src.data_utils import (
    label_to_index,
    load_annotation_csv,
    open_rgb_image,
    validate_research_splits,
    sha256_file,
    stable_rows_sha256,
)
from src.inference import EvidentialInferenceEngine
from experiment_utils import (
    build_experiment_manifest,
    cache_fingerprint,
    load_feature_cache,
    save_feature_cache,
    write_json_atomic,
)

RAW_FEATURE_ORDER: Tuple[str, ...] = FEATURE_NAMES
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "clip_only": ("one_minus_global_clip_similarity",),
    "sinkhorn_plus_clip": (
        "sinkhorn_divergence",
        "one_minus_global_clip_similarity",
    ),
}

# Quick profile checks every requested distortion family, including both
# darker and brighter images. Full profile adds milder/stronger levels.
QUICK_DISTORTIONS: List[Dict[str, Any]] = [
    {"name": "gaussian_blur_r1_5", "kind": "gaussian_blur", "radius": 1.5},
    {"name": "brightness_0_7", "kind": "brightness", "factor": 0.7},
    {"name": "brightness_1_3", "kind": "brightness", "factor": 1.3},
    {"name": "gaussian_noise_sigma15", "kind": "gaussian_noise", "sigma": 15.0},
    {"name": "jpeg_quality35", "kind": "jpeg", "quality": 35},
]
FULL_DISTORTIONS: List[Dict[str, Any]] = [
    {"name": "gaussian_blur_r0_8", "kind": "gaussian_blur", "radius": 0.8},
    {"name": "gaussian_blur_r1_5", "kind": "gaussian_blur", "radius": 1.5},
    {"name": "brightness_0_7", "kind": "brightness", "factor": 0.7},
    {"name": "brightness_1_3", "kind": "brightness", "factor": 1.3},
    {"name": "gaussian_noise_sigma10", "kind": "gaussian_noise", "sigma": 10.0},
    {"name": "gaussian_noise_sigma15", "kind": "gaussian_noise", "sigma": 15.0},
    {"name": "gaussian_noise_sigma25", "kind": "gaussian_noise", "sigma": 25.0},
    {"name": "jpeg_quality60", "kind": "jpeg", "quality": 60},
    {"name": "jpeg_quality35", "kind": "jpeg", "quality": 35},
    {"name": "jpeg_quality25", "kind": "jpeg", "quality": 25},
]


def select_cols(X: np.ndarray, subset: Iterable[str]) -> np.ndarray:
    return X[:, [RAW_FEATURE_ORDER.index(name) for name in subset]]


def apply_distortion(image: Image.Image, spec: Dict[str, Any], seed: int) -> Image.Image:
    """Apply one deterministic PIL/NumPy corruption and return RGB PIL image."""
    image = image.convert("RGB")
    kind = spec["kind"]

    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(spec["radius"])))

    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(spec["factor"]))

    if kind == "gaussian_noise":
        pixels = np.asarray(image, dtype=np.float32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, float(spec["sigma"]), size=pixels.shape)
        noisy = np.clip(pixels + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(spec["quality"]),
            subsampling=0,
            optimize=True,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    raise ValueError(f"unsupported distortion kind: {kind!r}")


def extract_features(
    engine: EvidentialInferenceEngine,
    csv_path: Path,
    rows: List[Dict[str, Any]],
    cache_path: Path,
    fingerprint: str,
    distortion: Dict[str, Any] | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load verified cached features or extract and cache a split."""
    if cache_path.is_file():
        X, y = load_feature_cache(cache_path, fingerprint)
        print(f"[cache] verified {cache_path} (X={X.shape})")
        return X, y

    X_values: List[List[float]] = []
    y_values: List[int] = []
    for index, row in enumerate(rows):
        image = open_rgb_image(Path(row["image_path"]))
        if distortion is not None:
            # Same row/seed produces the same underlying noise field across OT modes.
            image = apply_distortion(image, distortion, seed=seed + index)

        try:
            output = engine.analyze_image(image, str(row["statement"]))
            values = [
                float(output["sinkhorn_divergence"]),
                float(output.get("discarded_mass", 0.0)),
                1.0 - float(output["global_clip_similarity"]),
            ]
            if not np.isfinite(values).all():
                raise ValueError("inference produced non-finite features")
        except Exception as exc:
            label = distortion["name"] if distortion is not None else "clean"
            raise RuntimeError(
                f"feature extraction failed for {label}, {csv_path.name} "
                f"row {index + 1}: {exc}"
            ) from exc

        X_values.append(values)
        y_values.append(label_to_index(str(row["label"])))
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            condition = distortion["name"] if distortion is not None else "clean"
            print(f"[extract] {condition} {csv_path.name}: {index + 1}/{len(rows)}")

    X = np.asarray(X_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.int64)
    if len(y) == 0:
        raise RuntimeError(f"no features extracted from {csv_path}")
    save_feature_cache(cache_path, X, y, fingerprint)
    return X, y


def get_metrics(
    artifact: Any,
    X: np.ndarray,
    y: np.ndarray,
    test_groups: List[str],
    alpha: float,
) -> Dict[str, Any]:
    """Evaluate with the artifact's existing threshold; never recalibrate."""
    prediction_sets = artifact.predict_set(X)
    pair_metrics = coverage_and_abstention(prediction_sets, y.tolist(), alpha)
    group_metrics = coverage_by_group(prediction_sets, y.tolist(), test_groups)
    return {
        "q_hat_used": float(artifact.q_hat),
        "empirical_coverage": float(pair_metrics["empirical_coverage"]),
        "singleton_rate": float(pair_metrics["singleton_rate"]),
        "selective_accuracy": (
            None if pair_metrics["selective_accuracy"] is None
            else float(pair_metrics["selective_accuracy"])
        ),
        "abstention_rate": float(pair_metrics["abstention_rate"]),
        "recording_group_coverage": group_metrics,
    }


def add_deltas(clean: Dict[str, Any], condition: Dict[str, Any]) -> Dict[str, Any]:
    metrics = (
        "empirical_coverage",
        "singleton_rate",
        "selective_accuracy",
        "abstention_rate",
    )
    deltas: Dict[str, float | None] = {}
    for metric in metrics:
        clean_value = clean[metric]
        condition_value = condition[metric]
        deltas[metric] = (
            None
            if clean_value is None or condition_value is None
            else float(condition_value - clean_value)
        )

    clean_group = clean["recording_group_coverage"]["simultaneous_group_coverage"]
    condition_group = condition["recording_group_coverage"]["simultaneous_group_coverage"]
    deltas["recording_group_coverage"] = float(condition_group - clean_group)
    return deltas


def fingerprint_for(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    engine: EvidentialInferenceEngine,
    extra_config: Dict[str, Any] | None = None,
) -> str:
    inference_config = dict(engine.feature_config)
    if extra_config is not None:
        inference_config["robustness_transform"] = extra_config
    return cache_fingerprint(
        csv_path=csv_path,
        model_name=args.model_name,
        model_revision=engine.model_revision,
        feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
        inference_config=inference_config,
        image_paths=[row["image_path"] for row in rows],
    )


def main() -> None:
    args = parse_args()
    split_csvs = {
        "fit": Path(args.fit_csv).expanduser().resolve(),
        "calibration": Path(args.calibration_csv).expanduser().resolve(),
        "test": Path(args.test_csv).expanduser().resolve(),
    }
    split_rows = {
        name: load_annotation_csv(path) for name, path in split_csvs.items()
    }
    split_audit = validate_research_splits(split_rows)
    print("[split-audit] source-recording disjointness verified")
    if split_audit["camera_id_is_recording_alias"]:
        print(
            "[split-audit] camera_id is derived from recording ID; "
            "do not claim unseen-physical-camera evaluation"
        )

    test_groups = [str(row["source_recording_id"]) for row in split_rows["test"]]
    expected_test_labels = np.asarray(
        [label_to_index(str(row["label"])) for row in split_rows["test"]],
        dtype=np.int64,
    )
    if not test_groups or any(not value.strip() for value in test_groups):
        raise ValueError("test split contains blank source_recording_id values")

    distortions = QUICK_DISTORTIONS if args.profile == "quick" else FULL_DISTORTIONS
    results: Dict[str, Any] = {}
    cache_paths: Dict[str, str] = {}

    for mode_name, force_full in (("full_ot", True), ("partial_ot", False)):
        print(f"\n{'=' * 72}\n[mode] {mode_name}\n{'=' * 72}")
        engine = EvidentialInferenceEngine(
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            force_full_ot=force_full,
        )
        mode_cache_dir = Path(args.cache_dir) / mode_name
        matrices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Clean fit/calibration/test caches reuse the existing experiment's
        # fingerprints because their inference configuration is unchanged.
        for split_name in ("fit", "calibration", "test"):
            csv_path = split_csvs[split_name]
            rows = split_rows[split_name]
            fingerprint = fingerprint_for(csv_path, rows, args, engine)
            cache_path = mode_cache_dir / f"{split_name}.{fingerprint[:20]}.npz"
            cache_paths[f"{mode_name}:{split_name}:clean"] = str(cache_path)
            matrices[split_name] = extract_features(
                engine, csv_path, rows, cache_path, fingerprint
            )

        X_fit, y_fit = matrices["fit"]
        X_cal, y_cal = matrices["calibration"]
        X_clean_test, y_clean_test = matrices["test"]
        if len(y_clean_test) != len(test_groups):
            raise ValueError("clean test features do not align with test CSV rows")
        if not np.array_equal(y_clean_test, expected_test_labels):
            raise ValueError("cached clean test labels do not match current test CSV")

        fit_groups = [str(row["source_recording_id"]) for row in split_rows["fit"]]
        calibration_groups = [
            str(row["source_recording_id"]) for row in split_rows["calibration"]
        ]

        # Each artifact is fitted/calibrated ONCE on clean data. The exact same
        # artifact (including q_hat) is used for clean and every corruption.
        for variant_name, feature_names in VARIANTS.items():
            artifact = fit_conformal(
                select_cols(X_fit, feature_names),
                y_fit,
                select_cols(X_cal, feature_names),
                y_cal,
                alpha=args.alpha,
                feature_names=feature_names,
                inference_config=engine.feature_config,
                fit_csv_sha256=sha256_file(split_csvs["fit"]),
                calibration_csv_sha256=sha256_file(split_csvs["calibration"]),
                cli_model_revision=engine.model_revision,
                feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
                fit_group_ids=fit_groups,
                calibration_group_ids=calibration_groups,
            )

            clean_metrics = get_metrics(
                artifact,
                select_cols(X_clean_test, feature_names),
                y_clean_test,
                test_groups,
                args.alpha,
            )
            mode_variant_key = f"{mode_name}__{variant_name}"
            condition_results: Dict[str, Any] = {}
            print(
                f"[clean] {mode_variant_key}: "
                f"q_hat={artifact.q_hat:.6f}, "
                f"coverage={clean_metrics['empirical_coverage']:.3f}, "
                f"singleton={clean_metrics['singleton_rate']:.3f}, "
                f"selective_acc={clean_metrics['selective_accuracy']}"
            )

            # Extract each corrupted test matrix once per OT mode; evaluate it
            # against both feature variants using the already-fitted artifacts.
            for distortion in distortions:
                condition_name = str(distortion["name"])
                fingerprint = fingerprint_for(
                    split_csvs["test"],
                    split_rows["test"],
                    args,
                    engine,
                    extra_config={
                        **distortion,
                        "seed": args.seed if distortion["kind"] == "gaussian_noise" else 42,
                    },
                )
                cache_path = (
                    mode_cache_dir
                    / "robustness"
                    / f"test_{condition_name}.{fingerprint[:20]}.npz"
                )
                cache_paths[f"{mode_name}:test:{condition_name}"] = str(cache_path)
                X_corrupt, y_corrupt = extract_features(
                    engine,
                    split_csvs["test"],
                    split_rows["test"],
                    cache_path,
                    fingerprint,
                    distortion=distortion,
                    seed=args.seed,
                )
                if not np.array_equal(y_corrupt, expected_test_labels):
                    raise ValueError(
                        f"{condition_name}: corrupted feature labels do not match test CSV"
                    )

                corrupted_metrics = get_metrics(
                    artifact,
                    select_cols(X_corrupt, feature_names),
                    y_corrupt,
                    test_groups,
                    args.alpha,
                )
                if not np.isclose(
                    corrupted_metrics["q_hat_used"],
                    clean_metrics["q_hat_used"],
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise RuntimeError("q_hat changed during robustness evaluation")
                condition_results[condition_name] = {
                    "distortion": distortion,
                    "metrics": corrupted_metrics,
                    "delta_vs_clean": add_deltas(clean_metrics, corrupted_metrics),
                    "q_hat_fixed": True,
                }
                print(
                    f"[robust] {mode_variant_key} / {condition_name}: "
                    f"coverage={corrupted_metrics['empirical_coverage']:.3f}, "
                    f"singleton={corrupted_metrics['singleton_rate']:.3f}, "
                    f"selective_acc={corrupted_metrics['selective_accuracy']}"
                )

            results[mode_variant_key] = {
                "mode": mode_name,
                "variant": variant_name,
                "features": list(feature_names),
                "calibration_unit": artifact.calibration_unit,
                "calibration_group_count": artifact.calibration_group_count,
                "q_hat_clean_calibration": float(artifact.q_hat),
                "fixed_q_hat_for_all_conditions": True,
                "clean_baseline": clean_metrics,
                "distortions": condition_results,
            }

    manifest = build_experiment_manifest(
        script_name=Path(__file__).name,
        model_name=args.model_name,
        model_revision=engine.model_revision,
        feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
        split_csvs=split_csvs,
        split_rows=split_rows,
        split_audit=split_audit,
        configuration={
            "alpha": args.alpha,
            "experiment": "fixed_threshold_image_robustness",
            "profile": args.profile,
            "distortions": distortions,
            "calibration_data": "clean_fit_and_clean_calibration_only",
            "q_hat_policy": "fit_once_on_clean_calibration_then_freeze_for_all_test_conditions",
            "bootstrap_seed_for_noise": args.seed,
            "cache_paths": cache_paths,
            "calibration_unit": "source_recording_id_max",
            "selected_rows_sha256": {
                name: stable_rows_sha256(rows) for name, rows in split_rows.items()
            },
        },
    )

    output = Path(args.output)
    write_json_atomic(
        output,
        {
            "alpha": args.alpha,
            "manifest": manifest,
            "results": results,
            "important_note": (
                "All conformal artifacts were fitted and calibrated on clean fit and "
                "calibration splits only. The same q_hat was reused for clean and "
                "distorted test images; no corrupted test data was used for recalibration. "
                "Results are robustness measurements on the held-out proxy dataset, "
                "not evidence of real-world forensic validity."
            ),
        },
    )
    print(f"\n[write] {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-distortion robustness with frozen clean conformal thresholds."
    )
    parser.add_argument("--fit-csv", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--output", default="reports/group_v3/image_robustness_quick.json"
    )
    parser.add_argument("--cache-dir", default=".cache/group_v3_ot")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    return parser.parse_args()


if __name__ == "__main__":
    main()
