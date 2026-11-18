"""Demonstrate that quality-aware partial OT responds to image degradation.

Runs the inference pipeline on the UCA test split under four levels of
Gaussian blur, and reports how `discarded_mass` (partial-OT's readout of
how much patch-token mass it declined to align) behaves as a function of
blur severity.

No training, no calibration. This is a diagnostic sensitivity check.

Usage:
    py ablate_degradation_sensitivity.py                        # 20 recording groups, all 4 levels, legacy trust
    py ablate_degradation_sensitivity.py --max-examples 0       # full test split
    py ablate_degradation_sensitivity.py --max-examples 50 --device cuda
    py ablate_degradation_sensitivity.py --trust-mode absolute --trust-tau 50 --trust-k 100 \\
        --output reports/degradation_sensitivity_absolute.csv

Artifacts:
    reports/degradation_sensitivity.csv
    reports/degradation_sensitivity.summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data_utils import load_annotation_csv, open_rgb_image, sha256_file
from src.inference import EvidentialInferenceEngine
from src.image_quality import compute_patch_sharpness


SEVERITIES = [
    ("clean",    0.0),
    ("mild",     1.5),
    ("moderate", 3.0),
    ("heavy",    6.0),
]


def blur_rgb(image: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0.0:
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred)


def fit_absolute_trust_parameters(fit_rows):
    """Fit one global tau/k from fit-split patch sharpness, never per image."""
    all_values = []
    for index, row in enumerate(fit_rows, 1):
        image = open_rgb_image(Path(row["image_path"]))
        all_values.extend(compute_patch_sharpness(image))
        if index % 250 == 0 or index == len(fit_rows):
            print(f"[trust-fit] processed {index}/{len(fit_rows)} fit images")
    values = np.asarray(all_values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("could not derive finite patch-sharpness statistics from fit split")
    tau = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    k = float(max((q75 - q25) / 2.0, 1e-6))
    return tau, k


def deterministic_sample(rows, max_examples: int, seed: int):
    """Sample recording groups, retaining every pair from each selected group."""
    group_ids = sorted({str(row["source_recording_id"]) for row in rows})
    if max_examples <= 0 or max_examples >= len(group_ids):
        return list(rows)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(group_ids, size=max_examples, replace=False).tolist())
    return [row for row in rows if str(row["source_recording_id"]) in chosen]


def paired_monotonicity_summary(records: list[dict]) -> dict:
    """Measure monotonicity per fixed image/statement pair across blur levels."""
    by_pair: dict[str, list[dict]] = {}
    by_recording: dict[str, list[str]] = {}
    for record in records:
        pair_id = str(record["pair_id"])
        by_pair.setdefault(pair_id, []).append(record)
        by_recording.setdefault(str(record["source_recording_id"]), []).append(pair_id)

    pair_results = {}
    for pair_id, pair_records in by_pair.items():
        ordered = sorted(pair_records, key=lambda item: item["sigma"])
        if len(ordered) != len(SEVERITIES) or len({item["sigma"] for item in ordered}) != len(SEVERITIES):
            continue
        discarded = np.asarray([item["discarded_mass"] for item in ordered], dtype=float)
        sinkhorn = np.asarray([item["sinkhorn_divergence"] for item in ordered], dtype=float)
        pair_results[pair_id] = {
            "discarded_non_decreasing": bool(np.all(np.diff(discarded) >= -1e-8)),
            "discarded_non_increasing": bool(np.all(np.diff(discarded) <= 1e-8)),
            "sinkhorn_non_decreasing": bool(np.all(np.diff(sinkhorn) >= -1e-8)),
            "sinkhorn_non_increasing": bool(np.all(np.diff(sinkhorn) <= 1e-8)),
        }

    recording_results = []
    for pair_ids in by_recording.values():
        results = [pair_results[pair_id] for pair_id in pair_ids if pair_id in pair_results]
        if len(results) == len(pair_ids) and results:
            recording_results.append({
                key: all(item[key] for item in results)
                for key in next(iter(results)).keys()
            })

    def fraction(items, key):
        return float(np.mean([item[key] for item in items])) if items else None

    return {
        "paired_image_statement_pair_count": len(pair_results),
        "recordings_with_complete_pairs": len(recording_results),
        "pair_level": {
            key: fraction(list(pair_results.values()), key)
            for key in (
                "discarded_non_decreasing", "discarded_non_increasing",
                "sinkhorn_non_decreasing", "sinkhorn_non_increasing",
            )
        },
        "recording_level_all_pairs": {
            key: fraction(recording_results, key)
            for key in (
                "discarded_non_decreasing", "discarded_non_increasing",
                "sinkhorn_non_decreasing", "sinkhorn_non_increasing",
            )
        },
    }


def main() -> None:
    args = parse_args()

    rows_all = load_annotation_csv(Path(args.test_csv))
    rows = deterministic_sample(rows_all, args.max_examples, args.sample_seed)
    selected_recording_count = len({str(row["source_recording_id"]) for row in rows})
    total_recording_count = len({str(row["source_recording_id"]) for row in rows_all})
    print(
        f"[data] selected {selected_recording_count} of {total_recording_count} recordings "
        f"({len(rows)} image/statement pairs) from {args.test_csv}"
    )
    print(f"[data] sample_seed={args.sample_seed}; severities={[name for name, _ in SEVERITIES]}")

    trust_parameter_source = "not_applicable"
    if args.trust_mode == "absolute":
        if args.trust_fit_csv:
            fit_rows = load_annotation_csv(Path(args.trust_fit_csv))
            tau, k = fit_absolute_trust_parameters(fit_rows)
            trust_parameter_source = "fit_split_patch_sharpness_median_and_iqr_over_2"
        elif args.trust_tau is not None and args.trust_k is not None:
            tau, k = float(args.trust_tau), float(args.trust_k)
            trust_parameter_source = "explicit_cli_constants"
        else:
            raise ValueError(
                "absolute trust requires either --trust-fit-csv or both --trust-tau and --trust-k; "
                "do not silently use default constants"
            )
        if not np.isfinite([tau, k]).all() or k <= 0:
            raise ValueError("absolute trust tau and k must be finite, with k > 0")
        os.environ["OT_TRUST_TAU"] = repr(tau)
        os.environ["OT_TRUST_K"] = repr(k)
    else:
        tau, k = None, None

    print(
        f"[config] trust_mode={args.trust_mode} tau={tau} k={k} "
        f"trust_parameter_source={trust_parameter_source}"
    )
    print(f"[plan] {len(rows) * len(SEVERITIES)} CLIP+OT runs expected")

    engine = EvidentialInferenceEngine(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        trust_mode=args.trust_mode,
    )

    records: list[dict] = []
    per_severity: dict[str, list[tuple[float, float]]] = {
        name: [] for name, _ in SEVERITIES
    }

    for i, row in enumerate(rows, 1):
        image_path = Path(row["image_path"])
        clean_image = open_rgb_image(image_path)
        for severity_name, sigma in SEVERITIES:
            degraded = blur_rgb(clean_image, sigma)
            try:
                out = engine.analyze_image(degraded, str(row["statement"]))
            except Exception as exc:
                raise RuntimeError(
                    f"degradation inference failed for {image_path.name} / {severity_name}: {exc}"
                ) from exc
            pair_identity = (
                str(row["source_recording_id"]) + "\0"
                + str(row.get("statement_source_recording_id", "")) + "\0"
                + str(row["label"]) + "\0" + str(row["statement"])
            ).encode("utf-8")
            record = {
                "source": image_path.name,
                "pair_id": hashlib.sha256(pair_identity).hexdigest(),
                "source_recording_id": str(row["source_recording_id"]),
                "statement_source_recording_id": row.get("statement_source_recording_id"),
                "label": row["label"],
                "severity": severity_name,
                "sigma": sigma,
                "trust_mode": args.trust_mode,
                "discarded_mass": float(out["discarded_mass"]),
                "transported_mass": 1.0 - float(out["discarded_mass"]),
                "sinkhorn_divergence": float(out["sinkhorn_divergence"]),
                "global_clip_similarity": float(out["global_clip_similarity"]),
                "laplacian_variance": float(
                    out["visual_quality_metrics"]["laplacian_variance"]
                ),
                "mean_abs_hf_residual": float(
                    out["visual_quality_metrics"]["mean_absolute_high_frequency_residual"]
                ),
            }
            records.append(record)
            per_severity[severity_name].append(
                (record["discarded_mass"], record["sinkhorn_divergence"])
            )
        if i % 5 == 0 or i == len(rows):
            print(f"[progress] {i}/{len(rows)} images")

    if not records:
        raise RuntimeError("no records were produced")

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, dict] = {}
    for name, pairs in per_severity.items():
        if not pairs:
            continue
        disc = [p[0] for p in pairs]
        sink = [p[1] for p in pairs]
        summary[name] = {
            "n": len(pairs),
            "discarded_mass_mean": statistics.mean(disc),
            "discarded_mass_stdev": statistics.stdev(disc) if len(disc) > 1 else 0.0,
            "discarded_mass_min": min(disc),
            "discarded_mass_max": max(disc),
            "sinkhorn_mean": statistics.mean(sink),
            "sinkhorn_stdev": statistics.stdev(sink) if len(sink) > 1 else 0.0,
        }

    # Report paired trends for each fixed image/statement pair. Each source
    # recording contributes two distinct pairs (MATCH and proxy mismatch), so
    # grouping only by recording would incorrectly combine different statements.
    paired_metrics = paired_monotonicity_summary(records)
    summary["_meta"] = {
        "test_csv": str(Path(args.test_csv).expanduser().resolve()),
        "test_csv_sha256": sha256_file(Path(args.test_csv)),
        "full_test_sample_count": len(rows_all),
        "full_test_recording_count": total_recording_count,
        "selected_sample_count": len(rows),
        "selected_recording_count": selected_recording_count,
        "sample_seed": args.sample_seed,
        "selected_source_recording_ids": sorted({str(row["source_recording_id"]) for row in rows}),
        "trust_mode": args.trust_mode,
        "trust_tau": tau,
        "trust_k": k,
        "trust_parameter_source": trust_parameter_source,
        "trust_fit_csv": str(Path(args.trust_fit_csv).expanduser().resolve()) if args.trust_fit_csv else None,
        "trust_fit_csv_sha256": sha256_file(Path(args.trust_fit_csv)) if args.trust_fit_csv else None,
        "model_name": args.model_name,
        "model_revision": engine.model_revision,
        "feature_schema_version": engine.FEATURE_SCHEMA_VERSION,
        "severities": [name for name, _ in SEVERITIES],
        "paired_monotonicity": paired_metrics,
        "interpretation": (
            "Controlled Gaussian-blur sensitivity on a deterministic sample. Paired monotonicity "
            "is calculated per fixed image/statement pair across all blur levels. This is a diagnostic "
            "stress test, not evidence of robustness to real CCTV degradation or semantic reliability."
        ),
    }

    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[write] {output_csv}")
    print(f"[write] {summary_path}")

    print()
    print(
        f"{'severity':<10}{'n':>6}{'disc_mean':>12}{'disc_std':>10}"
        f"{'sink_mean':>12}{'sink_std':>10}"
    )
    print("-" * 60)
    prev_disc = None
    monotone = True
    for name, _ in SEVERITIES:
        s = summary.get(name)
        if s is None:
            continue
        marker = ""
        if prev_disc is not None and s["discarded_mass_mean"] < prev_disc - 1e-6:
            marker = "  <- non-monotone!"
            monotone = False
        print(
            f"{name:<10}{s['n']:>6}"
            f"{s['discarded_mass_mean']:>12.4f}"
            f"{s['discarded_mass_stdev']:>10.4f}"
            f"{s['sinkhorn_mean']:>12.4f}"
            f"{s['sinkhorn_stdev']:>10.4f}{marker}"
        )
        prev_disc = s["discarded_mass_mean"]

    print()
    if monotone:
        print("[OK] mean discarded_mass is non-decreasing across blur levels in this sample.")
    else:
        print("[!!] mean discarded_mass is NOT monotone across blur levels in this sample.")
    print("     Per-pair monotonicity fractions are recorded in the summary JSON; do not infer per-pair behaviour from aggregate means.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-csv", default="datasets/uca/test.csv")
    parser.add_argument(
        "--max-examples", type=int, default=20,
        help="0 = full test split; otherwise sample this many complete recording groups",
    )
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for reproducible degradation-study sampling")
    parser.add_argument("--output", default="reports/degradation_sensitivity.csv")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--trust-mode",
        default="heterogeneity",
        choices=("heterogeneity", "absolute"),
        help=(
            "'heterogeneity' = legacy sharpness/max normalization "
            "(inverts the degradation signal under monotone blur); "
            "'absolute' = sigmoid((sharpness - tau) / k), monotone in "
            "per-patch sharpness. Compare both and report both outcomes."
        ),
    )
    parser.add_argument(
        "--trust-fit-csv", default=None,
        help="Fit tau/k once from this fit-split CSV's patch-sharpness distribution",
    )
    parser.add_argument(
        "--trust-tau", type=float, default=None,
        help="Explicit sigmoid midpoint; must be paired with --trust-k if no --trust-fit-csv is supplied",
    )
    parser.add_argument(
        "--trust-k", type=float, default=None,
        help="Explicit positive sigmoid temperature used with --trust-tau.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()