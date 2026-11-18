"""Ablate full OT vs partial (quality-weighted) OT.

Tests the original claim of docs/PARTIAL_OT.md: does the trust-weighted
partial-OT mechanism improve downstream decision quality over full OT?

Full OT:  force visual_trust = text_trust = 1 (uniform marginals, no dustbin)
Partial:  visual_trust derived from per-patch Laplacian variance

Two variants compared per mode:
    - clip_only:        (1 - clip_similarity,)                -- 1D
    - sinkhorn_plus_clip: (sinkhorn, 1 - clip_similarity)     -- 2D, same features in both modes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.conformal import FEATURE_NAMES, coverage_and_abstention, coverage_by_group, fit_conformal
from src.data_utils import (
    label_to_index, load_annotation_csv, open_rgb_image, validate_research_splits, sha256_file,
    stable_rows_sha256,
)
from src.inference import EvidentialInferenceEngine
from experiment_utils import (
    build_experiment_manifest, cache_fingerprint, load_feature_cache,
    save_feature_cache, write_json_atomic,
)

RAW_FEATURE_ORDER: Tuple[str, ...] = FEATURE_NAMES

VARIANTS: Dict[str, Tuple[str, ...]] = {
    "clip_only": ("one_minus_global_clip_similarity",),
    "sinkhorn_plus_clip": ("sinkhorn_divergence", "one_minus_global_clip_similarity"),
}


def extract_raw(engine, csv_path: Path, cache_path: Path, fingerprint: str, rows=None):
    if cache_path.is_file():
        X, y = load_feature_cache(cache_path, fingerprint)
        print(f"[cache] verified {cache_path} (X={X.shape})")
        return X, y
    if rows is None:
        rows = load_annotation_csv(csv_path)
    X, y = [], []
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
                raise ValueError("inference produced non-finite features")
        except Exception as exc:
            raise RuntimeError(f"feature extraction failed at {csv_path.name} row {i}: {exc}") from exc
        X.append(values)
        y.append(label_to_index(str(row["label"])))
        if i % 1000 == 0 or i == len(rows):
            print(f"[extract] {csv_path.name}: {i}/{len(rows)}")
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    if not len(ya):
        raise RuntimeError(f"no features extracted from {csv_path}")
    save_feature_cache(cache_path, Xa, ya, fingerprint)
    return Xa, ya


def select_cols(X, subset):
    return X[:, [RAW_FEATURE_ORDER.index(n) for n in subset]]



def paired_recording_bootstrap(
    full_rows: List[Dict],
    partial_rows: List[Dict],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict:
    """Paired cluster bootstrap over complete source recordings.

    Each input row contains row_index, group_id, label, and prediction_set.
    Full and partial OT are evaluated on the same resampled recording IDs.
    Differences are Full OT minus Partial OT.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    def group_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            group_id = str(row["group_id"]).strip()
            if not group_id:
                raise ValueError("recording group IDs cannot be blank")
            grouped.setdefault(group_id, []).append(row)
        return grouped

    full_groups = group_rows(full_rows)
    partial_groups = group_rows(partial_rows)

    if not full_groups or set(full_groups) != set(partial_groups):
        raise ValueError(
            "Full and partial OT must contain the same non-empty recording groups"
        )

    group_ids = sorted(full_groups)
    for group_id in group_ids:
        full_items = sorted(full_groups[group_id], key=lambda item: int(item["row_index"]))
        partial_items = sorted(partial_groups[group_id], key=lambda item: int(item["row_index"]))
        full_identity = [
            (int(item["row_index"]), int(item["label"])) for item in full_items
        ]
        partial_identity = [
            (int(item["row_index"]), int(item["label"])) for item in partial_items
        ]
        if full_identity != partial_identity:
            raise ValueError(
                f"Full and partial rows are not aligned for recording {group_id}"
            )
        full_groups[group_id] = full_items
        partial_groups[group_id] = partial_items

    def calculate_metrics(sampled_group_ids: List[str], grouped_rows: Dict[str, List[Dict]]) -> Dict[str, float | None]:
        sampled_rows: List[Dict] = []
        group_covered: List[bool] = []

        for group_id in sampled_group_ids:
            rows = grouped_rows[group_id]
            sampled_rows.extend(rows)
            group_covered.append(
                all(int(row["label"]) in row["prediction_set"] for row in rows)
            )

        total = len(sampled_rows)
        if total == 0:
            raise ValueError("bootstrap sample contains no rows")

        covered = sum(
            int(row["label"]) in row["prediction_set"] for row in sampled_rows
        )
        singleton_rows = [
            row for row in sampled_rows if len(row["prediction_set"]) == 1
        ]
        singleton_correct = sum(
            int(row["prediction_set"][0]) == int(row["label"])
            for row in singleton_rows
        )

        return {
            "coverage": covered / total,
            "singleton_rate": len(singleton_rows) / total,
            "selective_accuracy": (
                singleton_correct / len(singleton_rows) if singleton_rows else None
            ),
            "recording_group_coverage": sum(group_covered) / len(group_covered),
        }

    # The un-resampled point estimates are reported alongside the intervals.
    observed_full = calculate_metrics(group_ids, full_groups)
    observed_partial = calculate_metrics(group_ids, partial_groups)

    rng = np.random.default_rng(seed)
    metric_names = (
        "coverage",
        "singleton_rate",
        "selective_accuracy",
        "recording_group_coverage",
    )
    differences = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        sampled = [
            str(value)
            for value in rng.choice(group_ids, size=len(group_ids), replace=True)
        ]
        full_metrics = calculate_metrics(sampled, full_groups)
        partial_metrics = calculate_metrics(sampled, partial_groups)

        for name in metric_names:
            full_value = full_metrics[name]
            partial_value = partial_metrics[name]
            # Selective accuracy is undefined if a resample has no singleton
            # predictions in either method; do not fabricate a value for it.
            if full_value is not None and partial_value is not None:
                differences[name].append(float(full_value - partial_value))

    summary: Dict[str, Dict] = {}
    for name in metric_names:
        values = np.asarray(differences[name], dtype=float)
        full_point = observed_full[name]
        partial_point = observed_partial[name]
        point_difference = (
            float(full_point - partial_point)
            if full_point is not None and partial_point is not None
            else None
        )
        if values.size == 0:
            summary[name] = {
                "full_ot_observed": full_point,
                "partial_ot_observed": partial_point,
                "observed_difference": point_difference,
                "valid_bootstrap_replicates": 0,
                "bootstrap_difference_mean": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
            }
            continue

        summary[name] = {
            "full_ot_observed": full_point,
            "partial_ot_observed": partial_point,
            "observed_difference": point_difference,
            "valid_bootstrap_replicates": int(values.size),
            "bootstrap_difference_mean": float(np.mean(values)),
            "ci_95_lower": float(np.percentile(values, 2.5)),
            "ci_95_upper": float(np.percentile(values, 97.5)),
        }

    return {
        "comparison": "full_ot_minus_partial_ot",
        "primary_variant": "sinkhorn_plus_clip",
        "bootstrap_unit": "source_recording_id",
        "recording_count": len(group_ids),
        "requested_bootstrap_replicates": n_bootstrap,
        "seed": seed,
        "interval_method": "paired_cluster_percentile_bootstrap",
        "metrics": summary,
        "interpretation": (
            "Intervals describe paired recording-level resampling uncertainty "
            "on this held-out proxy dataset. They do not establish forensic "
            "validity or generalization to real-world contradictions. For "
            "selective accuracy, replicates with no singleton predictions in "
            "either method are excluded."
        ),
    }

def main():
    args = parse_args()
    split_csvs = {
        "fit": Path(args.fit_csv).expanduser().resolve(),
        "calibration": Path(args.calibration_csv).expanduser().resolve(),
        "test": Path(args.test_csv).expanduser().resolve(),
    }
    split_rows = {name: load_annotation_csv(path) for name, path in split_csvs.items()}
    split_audit = validate_research_splits(split_rows)
    print("[split-audit] source-recording disjointness verified")
    if split_audit["camera_id_is_recording_alias"]:
        print("[split-audit] camera_id is derived from recording ID; do not claim unseen-physical-camera evaluation")

    results: Dict[str, Dict] = {}
    cache_paths: Dict[str, str] = {}
    row_predictions: Dict[str, List[Dict]] = {}
    for mode_name, force_full in {"full_ot": True, "partial_ot": False}.items():
        print(f"\n{'='*70}\n[mode] {mode_name}\n{'='*70}")
        engine = EvidentialInferenceEngine(
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            force_full_ot=force_full,
        )
        mode_cache_dir = Path(args.cache_dir) / mode_name
        inference_config = dict(engine.feature_config)
        matrices = {}
        for split_name, csv_path in split_csvs.items():
            fingerprint = cache_fingerprint(
                csv_path=csv_path,
                model_name=args.model_name,
                model_revision=engine.model_revision,
                feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
                inference_config=inference_config,
                image_paths=[row["image_path"] for row in split_rows[split_name]],
            )
            cache_path = mode_cache_dir / f"{split_name}.{fingerprint[:20]}.npz"
            cache_paths[f"{mode_name}:{split_name}"] = str(cache_path)
            matrices[split_name] = extract_raw(
                engine, csv_path, cache_path, fingerprint, rows=split_rows[split_name],
            )

        X_fit, y_fit = matrices["fit"]
        X_cal, y_cal = matrices["calibration"]
        X_test, y_test = matrices["test"]
        fit_groups = [str(row["source_recording_id"]) for row in split_rows["fit"]]
        cal_groups = [str(row["source_recording_id"]) for row in split_rows["calibration"]]
        test_groups = [str(row["source_recording_id"]) for row in split_rows["test"]]

        if len(y_test) != len(test_groups):
            raise ValueError("Test feature count does not match test recording metadata")
        expected_test_labels = np.asarray(
            [label_to_index(str(row["label"])) for row in split_rows["test"]],
            dtype=np.int64,
        )
        if not np.array_equal(y_test, expected_test_labels):
            raise ValueError("Cached test labels do not match the current test CSV")

        print(f"[stats] sinkhorn mean={X_fit[:,0].mean():.4f} std={X_fit[:,0].std():.4f}")
        print(f"[stats] discarded mean={X_fit[:,1].mean():.4f} std={X_fit[:,1].std():.4f}")

        for vname, cols in VARIANTS.items():
            art = fit_conformal(
                select_cols(X_fit, cols), y_fit,
                select_cols(X_cal, cols), y_cal,
                alpha=args.alpha,
                feature_names=cols,
                inference_config=engine.feature_config,
                fit_csv_sha256=sha256_file(split_csvs["fit"]),
                calibration_csv_sha256=sha256_file(split_csvs["calibration"]),
                cli_model_revision=engine.model_revision,
                feature_schema_version=engine.FEATURE_SCHEMA_VERSION,
                fit_group_ids=fit_groups,
                calibration_group_ids=cal_groups,
            )
            preds = art.predict_set(select_cols(X_test, cols))
            key = f"{mode_name}__{vname}"
            row_predictions[key] = [
                {
                    "row_index": int(index),
                    "group_id": str(group_id),
                    "label": int(label),
                    "prediction_set": [int(value) for value in prediction_set],
                }
                for index, (group_id, label, prediction_set) in enumerate(
                    zip(test_groups, y_test.tolist(), preds)
                )
            ]
            metrics = coverage_and_abstention(preds, y_test.tolist(), args.alpha)
            group_metrics = coverage_by_group(preds, y_test.tolist(), test_groups)
            results[key] = {
                "mode": mode_name,
                "variant": vname,
                "features": list(cols),
                "q_hat": art.q_hat,
                "calibration_unit": art.calibration_unit,
                "calibration_group_count": art.calibration_group_count,
                "empirical_coverage": metrics["empirical_coverage"],
                "singleton_rate": metrics["singleton_rate"],
                "selective_accuracy": metrics["selective_accuracy"],
                "abstention_rate": metrics["abstention_rate"],
                "recording_group_coverage": group_metrics,
            }
            print(f"  {vname:<22} cover={metrics['empirical_coverage']:.3f} "
                  f"single={metrics['singleton_rate']:.3f} "
                  f"sel_acc={metrics['selective_accuracy'] if metrics['selective_accuracy'] is not None else float('nan'):.3f} "
                  f"recording_cov={group_metrics['simultaneous_group_coverage']:.3f}")

    # Compare the primary Sinkhorn+CLIP variant with paired recording-level resampling.
    bootstrap_results = paired_recording_bootstrap(
        full_rows=row_predictions["full_ot__sinkhorn_plus_clip"],
        partial_rows=row_predictions["partial_ot__sinkhorn_plus_clip"],
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )

    # One shared manifest records the identical data partitions and both OT modes.
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
            "comparison": "full_ot_vs_quality_weighted_partial_ot",
            "cache_paths": cache_paths,
            "calibration_unit": "source_recording_id_max",
            "bootstrap_unit": "source_recording_id",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.seed,
            "selected_rows_sha256": {name: stable_rows_sha256(rows) for name, rows in split_rows.items()},
        },
    )
    output = Path(args.output)
    write_json_atomic(
        output,
        {
            "alpha": args.alpha,
            "manifest": manifest,
            "results": results,
            "paired_recording_bootstrap": bootstrap_results,
        },
    )
    print(f"\n[write] {output}")

    print(f"\n{'comparison':<40}{'cover':>8}{'single':>8}{'sel_acc':>10}")
    print("-" * 66)
    for key, report in results.items():
        print(f"{key:<40}{report['empirical_coverage']:>8.3f}"
              f"{report['singleton_rate']:>8.3f}{report['selective_accuracy'] if report['selective_accuracy'] is not None else float('nan'):>10.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-csv", required=True)
    p.add_argument("--calibration-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output", default="reports/full_vs_partial_ot.json")
    p.add_argument("--cache-dir", default=".cache/ot_ablation")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    main()