"""Ablate the conformal nonconformity score on a fixed split.

Six variants, same fit/calibration/test CSVs, same alpha. The question is
whether the OT-derived features contribute to decision quality versus the
trivial CLIP-similarity baseline.

This script does NOT train a neural network. "Fitting" a conformal artifact
means estimating a per-class Gaussian density in closed form (mean and
covariance), not running gradient descent.

Usage:
    py ablate_conformal_scores.py \\
        --fit-csv datasets/uca/fit.csv \\
        --calibration-csv datasets/uca/calibration.csv \\
        --test-csv datasets/uca/test.csv \\
        --alpha 0.10 \\
        --output reports/uca_ablation.json

On the first run, CLIP features are extracted for every example and cached
under --cache-dir. Subsequent runs reuse the cache and skip inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.conformal import (
    FEATURE_NAMES,
    coverage_and_abstention,
    coverage_by_group,
    fit_conformal,
)
from src.data_utils import label_to_index, load_annotation_csv, open_rgb_image, validate_research_splits, sha256_file, stable_rows_sha256
from src.inference import EvidentialInferenceEngine
from experiment_utils import (
    build_experiment_manifest, cache_fingerprint, load_feature_cache,
    save_feature_cache, write_json_atomic,
)

# Canonical column order for the cached raw-feature matrix.
RAW_FEATURE_ORDER: Tuple[str, ...] = FEATURE_NAMES

# Six ablations. Each name maps to the subset of RAW_FEATURE_ORDER
# (and of FEATURE_NAMES) that this variant uses.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "clip_only": (
        "one_minus_global_clip_similarity",
    ),
    "sinkhorn_only": (
        "sinkhorn_divergence",
    ),
    "discarded_only": (
        "discarded_mass",
    ),
    "sinkhorn_plus_clip": (
        "sinkhorn_divergence",
        "one_minus_global_clip_similarity",
    ),
    "discarded_plus_clip": (
        "discarded_mass",
        "one_minus_global_clip_similarity",
    ),
    "all_three": FEATURE_NAMES,
}


def _cache_key(cache_dir: Path, name: str, fingerprint: str) -> Path:
    return cache_dir / f"{name}.{fingerprint[:20]}.npz"


def extract_raw_features(
    engine: EvidentialInferenceEngine,
    csv_path: Path,
    cache_path: Path,
    fingerprint: str,
    *,
    rows: List[Dict[str, str | Path]] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with a cache bound to exact CSV/model/inference inputs."""
    if cache_path.is_file():
        X, y = load_feature_cache(cache_path, fingerprint)
        print(f"[cache] verified {cache_path} (X={X.shape})")
        return X, y

    if rows is None:
        rows = load_annotation_csv(csv_path)
    X_list: List[List[float]] = []
    y_list: List[int] = []
    for i, row in enumerate(rows, 1):
        image = open_rgb_image(Path(row["image_path"]))
        try:
            out = engine.analyze_image(image, str(row["statement"]))
            values = [
                float(out["sinkhorn_divergence"]),
                float(out.get("discarded_mass", 0.0)),
                1.0 - float(out["global_clip_similarity"]),
            ]
            if not np.isfinite(values).all():
                raise ValueError("inference produced non-finite conformal features")
        except Exception as exc:
            # Fail closed: silently dropping failed rows changes the calibration
            # population and invalidates the experiment manifest.
            raise RuntimeError(f"feature extraction failed at {csv_path.name} row {i}: {exc}") from exc
        X_list.append(values)
        y_list.append(label_to_index(str(row["label"])))
        if i % 1000 == 0 or i == len(rows):
            print(f"[extract] {csv_path.name}: {i}/{len(rows)}")

    if not X_list:
        raise RuntimeError(f"no features could be extracted from {csv_path}")
    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    save_feature_cache(cache_path, X, y, fingerprint)
    print(f"[cache] wrote {cache_path} (X={X.shape})")
    return X, y


def select_columns(X: np.ndarray, subset: Tuple[str, ...]) -> np.ndarray:
    idx = [RAW_FEATURE_ORDER.index(name) for name in subset]
    return X[:, idx]


def run_variant(
    X_fit: np.ndarray, y_fit: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    fit_group_ids: List[str], calibration_group_ids: List[str],
    test_group_ids: List[str],
    subset: Tuple[str, ...],
    alpha: float,
    inference_config: Dict[str, object],
    fit_csv_sha256: str,
    calibration_csv_sha256: str,
    model_revision: str,
    feature_schema_version: int,
):
    artifact = fit_conformal(
        select_columns(X_fit, subset), y_fit,
        select_columns(X_cal, subset), y_cal,
        alpha=alpha,
        feature_names=subset,
        inference_config=inference_config,
        fit_csv_sha256=fit_csv_sha256,
        calibration_csv_sha256=calibration_csv_sha256,
        cli_model_revision=model_revision,
        feature_schema_version=feature_schema_version,
        fit_group_ids=fit_group_ids,
        calibration_group_ids=calibration_group_ids,
    )
    prediction_sets = artifact.predict_set(select_columns(X_test, subset))
    metrics = coverage_and_abstention(prediction_sets, y_test.tolist(), alpha)
    group_metrics = coverage_by_group(prediction_sets, y_test.tolist(), test_group_ids)
    return artifact, metrics, group_metrics


def _select_rows(rows, max_examples: int | None, seed: int):
    """Select a deterministic sample of recordings, retaining every pair per recording."""
    if max_examples is None or max_examples <= 0:
        return list(rows)
    groups = sorted({str(row["source_recording_id"]) for row in rows})
    if max_examples >= len(groups):
        return list(rows)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(groups, size=max_examples, replace=False).tolist())
    return [row for row in rows if str(row["source_recording_id"]) in chosen]


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_csvs = {
        "fit": Path(args.fit_csv).expanduser().resolve(),
        "calibration": Path(args.calibration_csv).expanduser().resolve(),
        "test": Path(args.test_csv).expanduser().resolve(),
    }
    split_rows = {name: load_annotation_csv(path) for name, path in split_csvs.items()}
    split_audit = validate_research_splits(split_rows)
    print("[split-audit] source-recording disjointness verified")
    if split_audit["camera_id_is_recording_alias"]:
        print("[split-audit] camera_id is a recording-derived alias; no physical-camera generalisation claim")
    env_overlaps = split_audit["pairwise_group_overlap_counts"]["environment_id"]
    if any(env_overlaps.values()):
        print("[split-audit] activity categories overlap across splits (expected for UCA)")

    selected_rows = {
        name: _select_rows(rows, args.max_examples, args.sample_seed)
        for name, rows in split_rows.items()
    }

    print(f"[engine] loading {args.model_name} on device={args.device}")
    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
    )
    print(f"[engine] resolved revision: {engine.model_revision}")

    inference_config = dict(engine.feature_config)
    features: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cache_paths = {}
    for name, path in split_csvs.items():
        fingerprint = cache_fingerprint(
            csv_path=path,
            model_name=args.model_name,
            model_revision=engine.model_revision,
            feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
            inference_config=inference_config,
            selection_config={"max_examples": args.max_examples, "sample_seed": args.sample_seed},
            image_paths=[row["image_path"] for row in selected_rows[name]],
        )
        cache_path = _cache_key(cache_dir, name, fingerprint)
        cache_paths[name] = str(cache_path)
        features[name] = extract_raw_features(
            engine, path, cache_path, fingerprint, rows=selected_rows[name],
        )

    X_fit, y_fit = features["fit"]
    X_cal, y_cal = features["calibration"]
    X_test, y_test = features["test"]
    fit_groups = [str(row["source_recording_id"]) for row in selected_rows["fit"]]
    cal_groups = [str(row["source_recording_id"]) for row in selected_rows["calibration"]]
    test_groups = [str(row["source_recording_id"]) for row in selected_rows["test"]]

    print(f"\n[data] fit={len(y_fit)} calibration={len(y_cal)} test={len(y_test)}")
    print(f"[data] recording groups: fit={len(set(fit_groups))} calibration={len(set(cal_groups))} test={len(set(test_groups))}")
    print(f"[data] label counts: fit={np.bincount(y_fit, minlength=2).tolist()} "
          f"calibration={np.bincount(y_cal, minlength=2).tolist()} "
          f"test={np.bincount(y_test, minlength=2).tolist()}")

    results: Dict[str, Dict] = {}
    artifacts: Dict[str, object] = {}
    for variant_name, subset in VARIANTS.items():
        print(f"\n[ablate] {variant_name}: features={subset}")
        artifact, metrics, group_metrics = run_variant(
            X_fit, y_fit, X_cal, y_cal, X_test, y_test,
            fit_groups, cal_groups, test_groups,
            subset=subset,
            alpha=args.alpha,
            inference_config=engine.feature_config,
            fit_csv_sha256=sha256_file(split_csvs["fit"]),
            calibration_csv_sha256=sha256_file(split_csvs["calibration"]),
            model_revision=engine.model_revision,
            feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
        )
        artifacts[variant_name] = artifact
        results[variant_name] = {
            "features": list(subset),
            "feature_count": len(subset),
            "q_hat": artifact.q_hat,
            "calibration_unit": artifact.calibration_unit,
            "calibration_group_count": artifact.calibration_group_count,
            "empirical_coverage": metrics["empirical_coverage"],
            "singleton_rate": metrics["singleton_rate"],
            "selective_accuracy": metrics["selective_accuracy"],
            "abstention_rate": metrics["abstention_rate"],
            "average_set_size": metrics["average_set_size"],
            "recording_group_coverage": group_metrics,
        }
        print(f"  {variant_name:<22} cover={metrics['empirical_coverage']:.3f} "
              f"single={metrics['singleton_rate']:.3f} "
              f"sel_acc={metrics['selective_accuracy'] if metrics['selective_accuracy'] is not None else float('nan'):.3f} "
              f"recording_cov={group_metrics['simultaneous_group_coverage']:.3f}")

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
            "inference_config": inference_config,
            "max_recordings_per_split": args.max_examples,
            "sample_seed": args.sample_seed,
            "calibration_unit": "source_recording_id_max",
            "cache_paths": cache_paths,
            "selected_rows_sha256": {name: stable_rows_sha256(rows) for name, rows in selected_rows.items()},
            "selected_recording_ids": {name: sorted({str(row["source_recording_id"]) for row in rows}) for name, rows in selected_rows.items()},
        },
    )
    output_path = Path(args.output)
    payload = {"alpha": args.alpha, "manifest": manifest, "variants": results}
    write_json_atomic(output_path, payload)
    print(f"\n[write] {output_path}")

    if args.save_artifacts:
        art_dir = output_path.parent / "ablation_artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        for name, art in artifacts.items():
            art.save(art_dir / f"{name}.json")
        print(f"[write] artifacts → {art_dir}")

    print()
    header = f"{'variant':<22}{'cover':>8}{'single':>8}{'sel_acc':>10}{'abstain':>10}{'set_size':>10}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        sel = r["selective_accuracy"]
        sel_str = f"{sel:.3f}" if sel is not None else "n/a"
        print(f"{name:<22}{r['empirical_coverage']:>8.3f}{r['singleton_rate']:>8.3f}{sel_str:>10}{r['abstention_rate']:>10.3f}{r['average_set_size']:>10.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-csv", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", default="reports/ablation.json")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--cache-dir", default=".cache/ablation_features")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-recordings", "--max-examples", dest="max_examples", type=int, default=None,
                        help="Deterministically sample N complete source recordings per split; omit for full splits")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed used for deterministic smoke-test recording sampling")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Write each variant's conformal artifact under reports/ablation_artifacts/")
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if args.max_examples is not None and 0 < args.max_examples < 30:
        parser.error(
            "--max-examples now samples complete recordings and must be >= 30 "
            "to satisfy the minimum fit/calibration class and group counts"
        )
    return args


if __name__ == "__main__":
    main()