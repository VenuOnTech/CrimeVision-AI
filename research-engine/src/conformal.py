"""Split conformal prediction for the frozen CrimeVision inference engine.

Implements the three-way split-conformal classifier (Vovk et al. 2005;
Angelopoulos & Bates 2021, §2.1) for the MATCH / CONTRADICTION decision.

The three disjoint splits are:

    fit set         -> fits the class-conditional Gaussian density models
    calibration set -> scores those frozen models, produces q_hat
    test set        -> evaluated once; must not be used for fitting or tuning

The fit and calibration sets MUST share no source_recording_id, because
in-sample scoring invalidates the exchangeability assumption required by the
finite-sample coverage guarantee.

Coverage guarantee (marginal, under exchangeability of calibration and test):

    P(Y in C(X)) >= 1 - alpha

This is a theorem, not an empirical claim. The density model affects
prediction-set efficiency (set size, abstention), not coverage validity.

What this guarantee does NOT provide:
- Class-conditional or per-camera coverage.
- A per-case correctness guarantee.
- Any abstention-rate bound. The system can return both classes and achieve
  perfect coverage with 100% abstention. Selective accuracy must be reported.
- Robustness to distribution shift (new camera, new environment, adversarial
  input).
- Forensic or legal admissibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CLASS_TO_INDEX = {"MATCH": 0, "CONTRADICTION": 1}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}

FEATURE_NAMES = (
    "sinkhorn_divergence",
    "discarded_mass",
    "one_minus_global_clip_similarity",
)

SCHEMA_VERSION = 3
_MIN_PER_CLASS = 30
_MIN_TOTAL = 60
_MIN_CALIBRATION_GROUPS = 30


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return value


def _validate_feature_names(feature_names: Iterable[str]) -> Tuple[str, ...]:
    """Return a validated, de-duplicated tuple of feature names.

    Subsets of FEATURE_NAMES are permitted for ablation experiments. The
    order is preserved because the density models are dimension-indexed.
    """
    names = tuple(str(n) for n in feature_names)
    if not names:
        raise ValueError("feature_names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"feature_names contains duplicates: {names}")
    unknown = set(names) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(
            f"incompatible feature names: {sorted(unknown)} are not in the allowed "
            f"set {FEATURE_NAMES}"
        )
    return names


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction.

    q = s_(ceil((n+1)(1-alpha))), 1-indexed, or +inf when the index exceeds n.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        raise ValueError("cannot compute a conformal quantile from zero scores")
    if not np.isfinite(scores).all():
        raise ValueError("conformal scores must be finite")
    ordered = np.sort(scores)
    index = int(np.ceil((n + 1) * (1.0 - _validate_alpha(alpha))))
    if index > n:
        return float("inf")
    return float(ordered[index - 1])


@dataclass
class ClassDensity:
    mean: np.ndarray
    inverse_covariance: np.ndarray
    log_determinant: float
    log_prior: float
    sample_count: int

    @classmethod
    def fit(cls, features: np.ndarray, prior: float, ridge: float = 1e-3) -> "ClassDensity":
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("class features must be 2-D with >= 2 rows")
        mean = features.mean(axis=0)
        centered = features - mean
        covariance = (centered.T @ centered) / max(1, features.shape[0] - 1)
        regularized = covariance + ridge * np.eye(covariance.shape[0], dtype=float)
        sign, log_det = np.linalg.slogdet(regularized)
        if sign <= 0 or not np.isfinite(log_det):
            raise ValueError("per-class covariance is not positive definite")
        return cls(
            mean=mean,
            inverse_covariance=np.linalg.inv(regularized),
            log_determinant=float(log_det),
            log_prior=float(prior),
            sample_count=int(features.shape[0]),
        )

    def log_likelihood(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        centered = features - self.mean
        quadratic = np.einsum("ij,jk,ik->i", centered, self.inverse_covariance, centered)
        log_norm = -0.5 * (features.shape[1] * np.log(2.0 * np.pi) + self.log_determinant)
        return log_norm - 0.5 * quadratic + self.log_prior


@dataclass
class ConformalArtifact:
    alpha: float
    q_hat: float
    feature_names: Tuple[str, ...]
    class_to_index: Dict[str, int]
    density_match: ClassDensity
    density_contradiction: ClassDensity
    fit_sample_count: int
    fit_class_counts: Dict[str, int]
    calibration_sample_count: int
    calibration_class_counts: Dict[str, int]
    calibration_scores: Tuple[float, ...]
    inference_config: Optional[Dict[str, Any]] = None
    fit_csv_sha256: Optional[str] = None
    calibration_csv_sha256: Optional[str] = None
    cli_model_revision: Optional[str] = None
    feature_schema_version: Optional[int] = None
    calibration_unit: str = "example"
    calibration_group_count: int = 0
    fit_group_ids: Tuple[str, ...] = ()
    calibration_group_ids: Tuple[str, ...] = ()

    def nonconformity_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        expected_dim = len(self.feature_names)
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                f"features must have shape (n, {expected_dim}) for feature_names={self.feature_names}; "
                f"got shape {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("features contain non-finite values")
        ll_match = self.density_match.log_likelihood(features)
        ll_contra = self.density_contradiction.log_likelihood(features)
        max_ll = np.maximum(ll_match, ll_contra)
        p_contra = np.exp(ll_contra - max_ll) / (
            np.exp(ll_match - max_ll) + np.exp(ll_contra - max_ll)
        )
        p_match = 1.0 - p_contra
        return np.stack([p_contra, p_match], axis=1)

    def predict_set(self, features: np.ndarray, q_hat: Optional[float] = None) -> List[List[int]]:
        threshold = self.q_hat if q_hat is None else float(q_hat)
        if not np.isfinite(threshold) and threshold != float("inf"):
            raise ValueError("q_hat must be finite or positive infinity")
        scores = self.nonconformity_scores(features)
        result: List[List[int]] = []
        for row in scores:
            included = [i for i, score in enumerate(row) if float(score) <= threshold]
            result.append(included)
        return result

    def with_alpha(self, alpha: float) -> "ConformalArtifact":
        """Recompute the finite-sample quantile from stored calibration-unit scores."""
        alpha = _validate_alpha(alpha)
        q_hat = _conformal_quantile(np.asarray(self.calibration_scores), alpha)
        return replace(self, alpha=alpha, q_hat=q_hat)

    def to_json(self) -> Dict[str, Any]:
        group_calibrated = self.calibration_unit == "source_recording_id_max"
        return {
            "schema_version": SCHEMA_VERSION,
            "method": "split_conformal_classification",
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "feature_names": list(self.feature_names),
            "class_to_index": self.class_to_index,
            "density_match": {
                "mean": self.density_match.mean.tolist(),
                "inverse_covariance": self.density_match.inverse_covariance.tolist(),
                "log_determinant": self.density_match.log_determinant,
                "log_prior": self.density_match.log_prior,
                "sample_count": self.density_match.sample_count,
            },
            "density_contradiction": {
                "mean": self.density_contradiction.mean.tolist(),
                "inverse_covariance": self.density_contradiction.inverse_covariance.tolist(),
                "log_determinant": self.density_contradiction.log_determinant,
                "log_prior": self.density_contradiction.log_prior,
                "sample_count": self.density_contradiction.sample_count,
            },
            "fit_sample_count": self.fit_sample_count,
            "fit_class_counts": self.fit_class_counts,
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_class_counts": self.calibration_class_counts,
            "calibration_scores": list(self.calibration_scores),
            "inference_config": self.inference_config,
            "fit_csv_sha256": self.fit_csv_sha256,
            "calibration_csv_sha256": self.calibration_csv_sha256,
            "cli_model_revision": self.cli_model_revision,
            "feature_schema_version": self.feature_schema_version,
            "calibration_unit": self.calibration_unit,
            "calibration_group_count": self.calibration_group_count,
            "fit_group_ids": list(self.fit_group_ids),
            "calibration_group_ids": list(self.calibration_group_ids),
            "coverage_guarantee": 1.0 - self.alpha,
            "coverage_scope": (
                "simultaneous_within_recording_group_under_exchangeability_of_recording_groups"
                if group_calibrated else "marginal_example_level_under_exchangeability_of_examples"
            ),
            "warning": (
                "Coverage is marginal at the declared calibration unit. It is not "
                "class-conditional or a correctness guarantee for an individual "
                "accepted prediction. It does not guarantee an abstention rate or "
                "robustness to distribution shift."
            ),
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "ConformalArtifact":
        required = {
            "schema_version", "method", "alpha", "q_hat", "feature_names",
            "class_to_index", "density_match", "density_contradiction",
            "fit_sample_count", "fit_class_counts",
            "calibration_sample_count", "calibration_class_counts",
            "calibration_scores", "inference_config", "fit_csv_sha256", "calibration_csv_sha256",
            "calibration_unit", "calibration_group_count",
            "fit_group_ids", "calibration_group_ids",
        }
        if "schema_version" not in payload:
            raise ValueError("conformal artifact is missing schema_version")
        if int(payload["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported conformal artifact schema_version={payload['schema_version']}; "
                f"this build requires schema_version={SCHEMA_VERSION}. "
                "Regenerate the artifact using the current fit/calibration workflow."
            )
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"conformal artifact is missing fields: {sorted(missing)}")
        if payload["method"] != "split_conformal_classification":
            raise ValueError("unsupported conformal method")
        if dict(payload["class_to_index"]) != CLASS_TO_INDEX:
            raise ValueError("conformal artifact has incompatible class mapping")

        feature_names = _validate_feature_names(payload["feature_names"])
        expected_dim = len(feature_names)
        for density_key in ("density_match", "density_contradiction"):
            mean_dim = len(payload[density_key]["mean"])
            inv_cov_shape = np.asarray(payload[density_key]["inverse_covariance"]).shape
            if mean_dim != expected_dim:
                raise ValueError(
                    f"{density_key}.mean has dimension {mean_dim} but feature_names "
                    f"has length {expected_dim}"
                )
            if inv_cov_shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"{density_key}.inverse_covariance has shape {inv_cov_shape} "
                    f"but feature_names has length {expected_dim}"
                )

        def _load_density(node: Dict[str, Any]) -> ClassDensity:
            return ClassDensity(
                mean=np.asarray(node["mean"], dtype=float),
                inverse_covariance=np.asarray(node["inverse_covariance"], dtype=float),
                log_determinant=float(node["log_determinant"]),
                log_prior=float(node["log_prior"]),
                sample_count=int(node["sample_count"]),
            )

        calibration_unit = str(payload["calibration_unit"])
        if calibration_unit not in {"example", "source_recording_id_max"}:
            raise ValueError(f"unsupported calibration_unit={calibration_unit!r}")
        calibration_scores = tuple(float(v) for v in payload["calibration_scores"])
        group_count = int(payload["calibration_group_count"])
        if calibration_unit == "source_recording_id_max" and len(calibration_scores) != group_count:
            raise ValueError("group-calibrated artifact score count must equal calibration_group_count")
        if not calibration_scores or not np.isfinite(np.asarray(calibration_scores)).all():
            raise ValueError("calibration_scores must be non-empty and finite")

        fit_group_ids = tuple(str(v) for v in payload["fit_group_ids"])
        calibration_group_ids = tuple(str(v) for v in payload["calibration_group_ids"])
        if set(fit_group_ids).intersection(calibration_group_ids):
            raise ValueError("artifact fit and calibration recording IDs overlap")
        if calibration_unit == "source_recording_id_max" and len(calibration_group_ids) != group_count:
            raise ValueError("calibration recording ID count does not match calibration_group_count")

        return cls(
            alpha=_validate_alpha(float(payload["alpha"])),
            q_hat=float(payload["q_hat"]),
            feature_names=feature_names,
            class_to_index=dict(payload["class_to_index"]),
            density_match=_load_density(payload["density_match"]),
            density_contradiction=_load_density(payload["density_contradiction"]),
            fit_sample_count=int(payload["fit_sample_count"]),
            fit_class_counts=dict(payload["fit_class_counts"]),
            calibration_sample_count=int(payload["calibration_sample_count"]),
            calibration_class_counts=dict(payload["calibration_class_counts"]),
            calibration_scores=calibration_scores,
            inference_config=(dict(payload["inference_config"]) if payload["inference_config"] is not None else None),
            fit_csv_sha256=payload.get("fit_csv_sha256"),
            calibration_csv_sha256=payload.get("calibration_csv_sha256"),
            cli_model_revision=payload.get("cli_model_revision"),
            feature_schema_version=payload.get("feature_schema_version"),
            calibration_unit=calibration_unit,
            calibration_group_count=group_count,
            fit_group_ids=fit_group_ids,
            calibration_group_ids=calibration_group_ids,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, indent=2)
            handle.flush()
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ConformalArtifact":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))


def extract_conformal_features(
    inference_output: Dict[str, Any],
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Build the conformal feature vector from an inference-output dict.

    The returned column order matches `feature_names`. When called without
    arguments, returns the canonical three-feature vector that existing
    callers expect. Pass an explicit subset (e.g. `("sinkhorn_divergence",)`)
    to build an ablated variant.
    """
    feature_names = _validate_feature_names(feature_names)
    try:
        raw = {
            "sinkhorn_divergence": float(inference_output["sinkhorn_divergence"]),
            "discarded_mass": float(inference_output.get("discarded_mass", 0.0)),
            "one_minus_global_clip_similarity": 1.0 - float(inference_output["global_clip_similarity"]),
        }
    except KeyError as exc:
        raise ValueError(f"inference output missing field for conformal feature: {exc}") from exc
    values = np.array([raw[name] for name in feature_names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("conformal feature vector contains non-finite values")
    return values


def fit_conformal(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    alpha: float,
    *,
    feature_names: Iterable[str] = FEATURE_NAMES,
    inference_config: Optional[Dict[str, Any]] = None,
    fit_csv_sha256: Optional[str] = None,
    calibration_csv_sha256: Optional[str] = None,
    cli_model_revision: Optional[str] = None,
    feature_schema_version: Optional[int] = None,
    fit_group_ids: Optional[Sequence[str]] = None,
    calibration_group_ids: Optional[Sequence[str]] = None,
) -> ConformalArtifact:
    """Fit class densities and calibrate at example or recording-group level.

    When `calibration_group_ids` is supplied, the nonconformity score for each
    recording is the maximum true-label score among all examples belonging to
    that recording. The resulting threshold targets simultaneous inclusion of
    all labelled examples within a new exchangeable recording group. This
    avoids treating correlated pairs from one source recording as independent
    calibration units.
    """
    feature_names = _validate_feature_names(feature_names)
    expected_dim = len(feature_names)

    fit_features = np.asarray(fit_features, dtype=float)
    fit_labels = np.asarray(fit_labels, dtype=np.int64)
    calibration_features = np.asarray(calibration_features, dtype=float)
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    alpha = _validate_alpha(alpha)

    if fit_features.ndim != 2 or fit_features.shape[0] != fit_labels.shape[0]:
        raise ValueError("fit features must be 2-D and match fit labels length")
    if calibration_features.ndim != 2 or calibration_features.shape[0] != calibration_labels.shape[0]:
        raise ValueError("calibration features must be 2-D and match calibration labels length")
    if fit_features.shape[0] < _MIN_TOTAL:
        raise ValueError(f"need >= {_MIN_TOTAL} fit examples, got {fit_features.shape[0]}")
    if calibration_features.shape[0] < _MIN_TOTAL:
        raise ValueError(f"need >= {_MIN_TOTAL} calibration examples, got {calibration_features.shape[0]}")
    if fit_features.shape[1] != expected_dim or calibration_features.shape[1] != expected_dim:
        raise ValueError(
            f"features must have {expected_dim} columns (feature_names={feature_names}); "
            f"got fit={fit_features.shape[1]}, calibration={calibration_features.shape[1]}"
        )
    if not np.isfinite(fit_features).all() or not np.isfinite(calibration_features).all():
        raise ValueError("features contain non-finite values")
    if set(int(v) for v in np.unique(fit_labels).tolist()) != {0, 1}:
        raise ValueError("fit labels must include both classes")
    if set(int(v) for v in np.unique(calibration_labels).tolist()) != {0, 1}:
        raise ValueError("calibration labels must include both classes")

    normalized_fit_groups: Tuple[str, ...] = ()
    normalized_calibration_groups: Tuple[str, ...] = ()
    if fit_group_ids is not None:
        if len(fit_group_ids) != len(fit_labels):
            raise ValueError("fit_group_ids must match fit sample count")
        normalized_fit_groups = tuple(str(v).strip() for v in fit_group_ids)
        if any(not value for value in normalized_fit_groups):
            raise ValueError("fit_group_ids cannot contain blank values")
    if calibration_group_ids is not None:
        if len(calibration_group_ids) != len(calibration_labels):
            raise ValueError("calibration_group_ids must match calibration sample count")
        normalized_calibration_groups = tuple(str(v).strip() for v in calibration_group_ids)
        if any(not value for value in normalized_calibration_groups):
            raise ValueError("calibration_group_ids cannot contain blank values")
        unique_calibration_groups = tuple(sorted(set(normalized_calibration_groups)))
        if len(unique_calibration_groups) < _MIN_CALIBRATION_GROUPS:
            raise ValueError(
                f"need >= {_MIN_CALIBRATION_GROUPS} distinct calibration groups, "
                f"got {len(unique_calibration_groups)}"
            )
    else:
        unique_calibration_groups = ()

    if normalized_fit_groups and normalized_calibration_groups:
        overlap = set(normalized_fit_groups).intersection(normalized_calibration_groups)
        if overlap:
            raise ValueError(f"fit/calibration group leakage: {sorted(overlap)[:5]}")

    fit_match = fit_labels == 0
    fit_contra = fit_labels == 1
    count_fit_match = int(fit_match.sum())
    count_fit_contra = int(fit_contra.sum())
    if count_fit_match < _MIN_PER_CLASS or count_fit_contra < _MIN_PER_CLASS:
        raise ValueError(
            f"need >= {_MIN_PER_CLASS} fit examples per class, got "
            f"MATCH={count_fit_match}, CONTRADICTION={count_fit_contra}"
        )

    prior_match = count_fit_match / float(len(fit_labels))
    prior_contra = count_fit_contra / float(len(fit_labels))
    density_match = ClassDensity.fit(fit_features[fit_match], prior=prior_match)
    density_contra = ClassDensity.fit(fit_features[fit_contra], prior=prior_contra)

    ll_match = density_match.log_likelihood(calibration_features)
    ll_contra = density_contra.log_likelihood(calibration_features)
    max_ll = np.maximum(ll_match, ll_contra)
    p_contra = np.exp(ll_contra - max_ll) / (
        np.exp(ll_match - max_ll) + np.exp(ll_contra - max_ll)
    )
    p_match = 1.0 - p_contra

    example_scores = np.empty(len(calibration_labels), dtype=float)
    example_scores[calibration_labels == 0] = p_contra[calibration_labels == 0]
    example_scores[calibration_labels == 1] = p_match[calibration_labels == 1]

    if normalized_calibration_groups:
        grouped_scores: Dict[str, float] = {}
        for group_id, score in zip(normalized_calibration_groups, example_scores):
            grouped_scores[group_id] = max(grouped_scores.get(group_id, float("-inf")), float(score))
        scores = np.asarray([grouped_scores[group] for group in sorted(grouped_scores)], dtype=float)
        calibration_unit = "source_recording_id_max"
        calibration_group_count = len(grouped_scores)
    else:
        scores = example_scores
        calibration_unit = "example"
        calibration_group_count = len(scores)

    q_hat = _conformal_quantile(scores, alpha)

    return ConformalArtifact(
        alpha=alpha,
        q_hat=q_hat,
        feature_names=feature_names,
        class_to_index=dict(CLASS_TO_INDEX),
        density_match=density_match,
        density_contradiction=density_contra,
        fit_sample_count=int(len(fit_labels)),
        fit_class_counts={"MATCH": count_fit_match, "CONTRADICTION": count_fit_contra},
        calibration_sample_count=int(len(calibration_labels)),
        calibration_class_counts={
            "MATCH": int((calibration_labels == 0).sum()),
            "CONTRADICTION": int((calibration_labels == 1).sum()),
        },
        calibration_scores=tuple(float(v) for v in scores),
        inference_config=dict(inference_config) if inference_config is not None else None,
        fit_csv_sha256=fit_csv_sha256,
        calibration_csv_sha256=calibration_csv_sha256,
        cli_model_revision=cli_model_revision,
        feature_schema_version=feature_schema_version,
        calibration_unit=calibration_unit,
        calibration_group_count=calibration_group_count,
        fit_group_ids=tuple(sorted(set(normalized_fit_groups))),
        calibration_group_ids=tuple(sorted(set(normalized_calibration_groups))),
    )


def prediction_set_to_assessment(prediction_set: Sequence[int]) -> Tuple[str, Optional[str]]:
    labels = sorted(set(int(v) for v in prediction_set))
    if labels == [0]:
        return "MATCH", None
    if labels == [1]:
        return "CONTRADICTION", None
    if labels == [0, 1]:
        return "INDETERMINATE", (
            "The conformal prediction set contains both MATCH and CONTRADICTION "
            "at the configured coverage level; the system abstains."
        )
    if labels == []:
        return "INDETERMINATE", (
            "The conformal prediction set is empty at the configured coverage "
            "level; the input is not well represented by the calibration data."
        )
    raise ValueError(f"unexpected conformal prediction set: {labels}")


def coverage_and_abstention(
    prediction_sets: Iterable[Sequence[int]],
    true_labels: Iterable[int],
    alpha: float,
) -> Dict[str, Any]:
    """Coverage, abstention, selective accuracy, and set size on a held-out set.

    Conformal guarantees only `P(Y in C(X)) >= 1 - alpha` (marginal). It does
    not guarantee selective accuracy, abstention rate, or class-conditional
    coverage. All are reported here so the tradeoff is visible.
    """
    alpha = _validate_alpha(alpha)
    sets = [sorted(int(v) for v in s) for s in prediction_sets]
    labels = [int(v) for v in true_labels]
    if len(sets) != len(labels):
        raise ValueError("prediction_sets and true_labels must have equal length")
    total = len(labels)
    if total == 0:
        raise ValueError("cannot compute coverage on an empty set")

    covered = sum(1 for s, y in zip(sets, labels) if y in s)
    singleton = sum(1 for s in sets if len(s) == 1)
    singleton_correct = sum(
        1 for s, y in zip(sets, labels) if len(s) == 1 and s[0] == y
    )
    empty = sum(1 for s in sets if len(s) == 0)
    both = sum(1 for s in sets if len(s) == 2)
    avg_set_size = sum(len(s) for s in sets) / total

    per_class: Dict[str, Any] = {}
    for class_index, class_name in INDEX_TO_CLASS.items():
        mask = [y == class_index for y in labels]
        n = sum(mask)
        if n == 0:
            continue
        class_covered = sum(
            1 for s, y, m in zip(sets, labels, mask) if m and y in s
        )
        class_singletons = sum(
            1 for s, y, m in zip(sets, labels, mask)
            if m and len(s) == 1
        )
        class_singleton_correct = sum(
            1 for s, y, m in zip(sets, labels, mask)
            if m and len(s) == 1 and s[0] == y
        )
        per_class[class_name] = {
            "sample_count": n,
            "coverage": class_covered / n,
            "singleton_rate": class_singletons / n,
            "selective_accuracy": (
                class_singleton_correct / class_singletons
                if class_singletons > 0 else None
            ),
        }

    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": covered / total,
        "sample_count": total,
        "singleton_count": singleton,
        "singleton_rate": singleton / total,
        "selective_accuracy": (
            singleton_correct / singleton if singleton > 0 else None
        ),
        "empty_count": empty,
        "empty_rate": empty / total,
        "both_classes_count": both,
        "both_classes_rate": both / total,
        "abstention_count": both + empty,
        "abstention_rate": (both + empty) / total,
        "average_set_size": avg_set_size,
        "per_class": per_class,
        "note": (
            "The theoretical guarantee is marginal coverage under "
            "exchangeability. Selective accuracy and abstention are empirical. "
            "Per-group coverage can fall below target."
        ),
    }


def coverage_by_group(
    prediction_sets: Iterable[Sequence[int]],
    true_labels: Iterable[int],
    group_ids: Iterable[str],
) -> Dict[str, Any]:
    """Report simultaneous true-label inclusion for each complete group."""
    sets = [set(int(v) for v in item) for item in prediction_sets]
    labels = [int(v) for v in true_labels]
    groups = [str(v) for v in group_ids]
    if not (len(sets) == len(labels) == len(groups)):
        raise ValueError("prediction_sets, true_labels and group_ids must have equal length")
    if not groups or any(not value for value in groups):
        raise ValueError("group_ids must be non-empty and contain no blank values")

    by_group: Dict[str, list[bool]] = {}
    for prediction_set, label, group in zip(sets, labels, groups):
        by_group.setdefault(group, []).append(label in prediction_set)
    covered_groups = sum(all(results) for results in by_group.values())
    group_count = len(by_group)
    return {
        "group_key": "source_recording_id",
        "group_count": group_count,
        "simultaneously_covered_group_count": covered_groups,
        "simultaneous_group_coverage": covered_groups / group_count,
        "groups_with_any_missed_label": group_count - covered_groups,
        "group_sizes": {"min": min(map(len, by_group.values())), "max": max(map(len, by_group.values()))},
        "interpretation": (
            "Descriptive held-out recording-level simultaneous coverage. The split-conformal "
            "guarantee applies only under exchangeability of recording groups and the same "
            "group construction protocol; it is not per-example conditional correctness."
        ),
    }
