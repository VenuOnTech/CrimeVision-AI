"""CLIP patch/token extraction and uncertainty-aware partial-OT inference.

No random or synthetic embeddings are created. If a trained evidential head,
a conformal artifact, or neither is configured, the engine returns
INDETERMINATE when appropriate while still exposing OT diagnostics and the
heatmap.

Decision mechanisms (mutually exclusive):
- Evidential head + calibration (requires supervised training).
- Conformal artifact (training-free, marginal coverage guarantee).
- Neither -> the engine always returns INDETERMINATE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .evidential_fusion import EvidentialOptimalTransport, EvidentialPairHead
from .image_quality import compute_image_quality_metrics, compute_patch_sharpness


class EvidentialInferenceEngine:
    FEATURE_SCHEMA_VERSION = 4
    QUALITY_FEATURE_NAMES = (
        "mean_absolute_high_frequency_residual",
        "laplacian_variance",
        "luminance_standard_deviation",
    )
    QUALITY_FEATURE_CLIP = 10.0
    CLASS_TO_INDEX = {"MATCH": 0, "CONTRADICTION": 1}
    PATCH_GRID = 7  # CLIP ViT-B/32: 224 / 32 = 7

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_revision: Optional[str] = None,
        device: Optional[str] = None,
        head_checkpoint_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        conformal_artifact_path: Optional[str] = None,
        epsilon: float = 0.1,
        sinkhorn_max_iterations: int = 2000,
        force_full_ot: bool = False,
        trust_mode: str = "heterogeneity",
    ) -> None:
        if trust_mode not in ("heterogeneity", "absolute"):
            raise ValueError(
                f"trust_mode must be 'heterogeneity' or 'absolute', got {trust_mode!r}"
            )
        self.force_full_ot = bool(force_full_ot)
        self.trust_mode = trust_mode
        self.ot_trust_tau = float(os.getenv("OT_TRUST_TAU", "50.0"))
        self.ot_trust_k = float(os.getenv("OT_TRUST_K", "100.0"))
        if self.trust_mode == "absolute":
            if not math.isfinite(self.ot_trust_tau) or not math.isfinite(self.ot_trust_k) or self.ot_trust_k <= 0:
                raise ValueError("absolute trust requires finite OT_TRUST_TAU and positive OT_TRUST_K")
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for CLIP inference. "
                "Install research-engine/requirements.txt."
            ) from exc

        requested_device = device or "auto"
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CLIP inference but is not available")
        self.device = torch.device(requested_device)
        self.model_name = model_name
        requested_revision = model_revision or os.getenv("CLIP_MODEL_REVISION") or "main"
        self.requested_model_revision = requested_revision
        self.processor = CLIPProcessor.from_pretrained(model_name, revision=requested_revision)
        self.model = CLIPModel.from_pretrained(model_name, revision=requested_revision).to(self.device)
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or requested_revision
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.ot_solver = EvidentialOptimalTransport(
            epsilon=epsilon,
            max_iterations=sinkhorn_max_iterations,
        )
        self.feature_config = {
            "trust_mode": self.trust_mode,
            "force_full_ot": self.force_full_ot,
            "sinkhorn_epsilon": float(self.ot_solver.solver.epsilon),
            "dustbin_cost": float(self.ot_solver.dustbin_cost),
        }
        if self.trust_mode == "absolute":
            self.feature_config.update({
                "ot_trust_tau": self.ot_trust_tau,
                "ot_trust_k": self.ot_trust_k,
            })
        self.head: Optional[EvidentialPairHead] = None
        self.calibration: Optional[Dict[str, Any]] = None
        self.checkpoint_sha256: Optional[str] = None
        self.quality_feature_mean: Optional[torch.Tensor] = None
        self.quality_feature_std: Optional[torch.Tensor] = None
        self.conformal: Optional[Any] = None

        if bool(head_checkpoint_path) != bool(calibration_path):
            raise ValueError(
                "Both head_checkpoint_path and calibration_path must be supplied together. "
                "A classifier without matching calibration is not allowed to issue decisions."
            )
        if conformal_artifact_path and (head_checkpoint_path or calibration_path):
            raise ValueError(
                "Configure either the evidential head "
                "(head_checkpoint_path + calibration_path) OR the conformal predictor "
                "(conformal_artifact_path), not both. They are independent "
                "decision mechanisms."
            )

        if head_checkpoint_path and calibration_path:
            self._load_trained_head(Path(head_checkpoint_path), Path(calibration_path))
        elif conformal_artifact_path:
            from .conformal import ConformalArtifact
            artifact = ConformalArtifact.load(Path(conformal_artifact_path))
            if (
                artifact.cli_model_revision is not None
                and artifact.cli_model_revision != self.model_revision
            ):
                raise ValueError(
                    "Conformal artifact was calibrated with a different CLIP "
                    f"revision ({artifact.cli_model_revision!r}) than the currently "
                    f"loaded backbone ({self.model_revision!r})."
                )
            if artifact.inference_config is not None and artifact.inference_config != self.feature_config:
                raise ValueError(
                    "Conformal artifact was calibrated with different OT/trust inference settings. "
                    f"Artifact={artifact.inference_config}, current={self.feature_config}."
                )
            if (
                artifact.feature_schema_version is not None
                and artifact.feature_schema_version != self.FEATURE_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Conformal artifact uses an incompatible feature schema version "
                    f"({artifact.feature_schema_version}) vs current "
                    f"({self.FEATURE_SCHEMA_VERSION})."
                )
            self.conformal = artifact

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_trained_head(self, checkpoint_path: Path, calibration_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evidential checkpoint not found: {checkpoint_path}")
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {calibration_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {
            "state_dict", "model_name", "model_revision", "input_dim", "class_to_index",
            "feature_schema_version", "quality_feature_names",
            "quality_feature_mean", "quality_feature_std", "quality_feature_clip_abs_z",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is missing metadata fields: {sorted(missing)}")
        if checkpoint["model_name"] != self.model_name:
            raise ValueError(
                f"Checkpoint model is {checkpoint['model_name']!r}, but inference uses {self.model_name!r}"
            )
        if checkpoint["model_revision"] != self.model_revision:
            raise ValueError(
                "Checkpoint CLIP revision does not match the currently loaded backbone. "
                f"Checkpoint={checkpoint['model_revision']!r}; loaded={self.model_revision!r}."
            )
        if int(checkpoint["input_dim"]) != EvidentialPairHead.INPUT_DIM:
            raise ValueError("Checkpoint input dimension does not match the current feature schema")
        if checkpoint["class_to_index"] != self.CLASS_TO_INDEX:
            raise ValueError("Checkpoint class mapping is incompatible with inference")
        if int(checkpoint["feature_schema_version"]) != self.FEATURE_SCHEMA_VERSION:
            raise ValueError("Checkpoint feature schema version is incompatible with inference")
        if tuple(checkpoint["quality_feature_names"]) != self.QUALITY_FEATURE_NAMES:
            raise ValueError("Checkpoint quality-feature schema is incompatible with inference")
        if float(checkpoint["quality_feature_clip_abs_z"]) != self.QUALITY_FEATURE_CLIP:
            raise ValueError("Checkpoint quality-feature clipping is incompatible with inference")

        quality_mean = torch.as_tensor(checkpoint["quality_feature_mean"], dtype=torch.float32)
        quality_std = torch.as_tensor(checkpoint["quality_feature_std"], dtype=torch.float32)
        if quality_mean.shape != (len(self.QUALITY_FEATURE_NAMES),) or quality_std.shape != quality_mean.shape:
            raise ValueError("Checkpoint quality-feature normalization has an invalid shape")
        if not torch.isfinite(quality_mean).all() or not torch.isfinite(quality_std).all() or torch.any(quality_std <= 0):
            raise ValueError("Checkpoint quality-feature normalization contains invalid values")
        self.quality_feature_mean = quality_mean.to(self.device)
        self.quality_feature_std = quality_std.to(self.device)

        self.head = EvidentialPairHead(input_dim=int(checkpoint["input_dim"])).to(self.device)
        self.head.load_state_dict(checkpoint["state_dict"], strict=True)
        self.head.eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        calibration_required = {
            "schema_version",
            "model_name",
            "model_revision",
            "checkpoint_sha256",
            "feature_schema_version",
            "contradiction_probability_threshold",
            "max_epistemic_uncertainty",
            "calibration_sample_count",
            "calibration_class_counts",
            "quality_feature_names",
            "quality_feature_clip_abs_z",
        }
        absent = calibration_required.difference(calibration)
        if absent:
            raise ValueError(f"Calibration artifact is missing fields: {sorted(absent)}")
        if int(calibration["schema_version"]) != 1:
            raise ValueError("Unsupported calibration artifact schema version")
        if calibration["model_name"] != self.model_name:
            raise ValueError("Calibration artifact was generated for a different CLIP model")
        if calibration["model_revision"] != self.model_revision:
            raise ValueError("Calibration artifact was generated for a different CLIP model revision")
        if calibration.get("class_to_index") != self.CLASS_TO_INDEX:
            raise ValueError("Calibration artifact has an incompatible class mapping")
        if int(calibration["feature_schema_version"]) != self.FEATURE_SCHEMA_VERSION:
            raise ValueError("Calibration artifact uses an incompatible feature schema")
        if tuple(calibration["quality_feature_names"]) != self.QUALITY_FEATURE_NAMES:
            raise ValueError("Calibration artifact has an incompatible quality-feature schema")
        if float(calibration["quality_feature_clip_abs_z"]) != self.QUALITY_FEATURE_CLIP:
            raise ValueError("Calibration artifact has incompatible quality-feature clipping")

        actual_hash = self._sha256_file(checkpoint_path)
        if calibration["checkpoint_sha256"] != actual_hash:
            raise ValueError("Calibration artifact does not match this exact checkpoint file")
        if int(calibration["calibration_sample_count"]) < 1:
            raise ValueError("Calibration artifact must contain at least one calibration example")

        probability_threshold = float(calibration["contradiction_probability_threshold"])
        uncertainty_threshold = float(calibration["max_epistemic_uncertainty"])
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError("Calibrated probability threshold must be within [0, 1]")
        if not 0.0 < uncertainty_threshold <= 1.0:
            raise ValueError("Calibrated uncertainty threshold must be in (0, 1]")

        self.checkpoint_sha256 = actual_hash
        self.calibration = calibration

    @classmethod
    def quality_metrics_to_tensor(cls, metrics: Dict[str, Any]) -> torch.Tensor:
        values = [float(metrics[name]) for name in cls.QUALITY_FEATURE_NAMES]
        tensor = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("Image-quality metrics contain NaN or infinite values")
        return tensor

    @classmethod
    def quality_features_in_training_domain(
        cls,
        quality_features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> bool:
        if quality_features.ndim != 2 or quality_features.shape[1] != len(cls.QUALITY_FEATURE_NAMES):
            raise ValueError("quality_features must have shape (batch, 3)")
        quality_features = quality_features.to(device=mean.device, dtype=mean.dtype)
        std = std.to(device=mean.device, dtype=mean.dtype)
        if not torch.isfinite(quality_features).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Quality features and normalization values must be finite")
        if torch.any(std <= 0):
            raise ValueError("Quality-feature standard deviations must be positive")
        raw_z = (quality_features - mean) / std
        return bool(torch.all(torch.abs(raw_z) <= cls.QUALITY_FEATURE_CLIP).item())

    @classmethod
    def standardize_quality_features(
        cls,
        quality_features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if quality_features.ndim != 2 or quality_features.shape[1] != len(cls.QUALITY_FEATURE_NAMES):
            raise ValueError("quality_features must have shape (batch, 3)")
        if mean.shape != (len(cls.QUALITY_FEATURE_NAMES),) or std.shape != mean.shape:
            raise ValueError("Quality normalization vectors must have shape (3,)")
        if not torch.isfinite(quality_features).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Quality features and normalization values must be finite")
        if torch.any(std <= 0):
            raise ValueError("Quality-feature standard deviations must be positive")
        quality_features = quality_features.to(device=mean.device, dtype=mean.dtype)
        std = std.to(device=mean.device, dtype=mean.dtype)
        return ((quality_features - mean) / std).clamp(
            min=-cls.QUALITY_FEATURE_CLIP,
            max=cls.QUALITY_FEATURE_CLIP,
        )

    def _extract_features(self, image: Image.Image, statement: str) -> Dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if image.width < 2 or image.height < 2:
            raise ValueError("image dimensions must be at least 2x2 pixels")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("A non-empty witness statement is required")
        if len(statement) > 10000:
            raise ValueError("Statement exceeds the 10,000-character limit")

        prepared_image = image.convert("RGB")
        inputs = self.processor(
            text=[statement.strip()],
            images=prepared_image,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        context_length = int(self.model.config.text_config.max_position_embeddings)
        if int(inputs["input_ids"].shape[1]) > context_length:
            raise ValueError(
                f"Witness statement exceeds CLIP's {context_length}-token context window. "
                "Split it into shorter evidence-linked statements; automatic truncation is disabled."
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        quality_metrics = compute_image_quality_metrics(prepared_image)
        quality_features = self.quality_metrics_to_tensor(quality_metrics)
        patch_sharpness = compute_patch_sharpness(prepared_image, grid=self.PATCH_GRID)
        sharpness_tensor = torch.tensor(patch_sharpness, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            vision_output = self.model.vision_model(pixel_values=inputs["pixel_values"])
            text_output = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

            vision_tokens = vision_output.last_hidden_state

            visual_patch_features = self.model.visual_projection(
                vision_tokens[:, 1:, :]
            )[0]

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs.get("attention_mask")
            valid_mask = (
                attention_mask[0].bool()
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )
            tokenizer = self.processor.tokenizer
            special_ids = {
                token_id
                for token_id in (
                    getattr(tokenizer, "bos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None),
                    getattr(tokenizer, "pad_token_id", None),
                )
                if token_id is not None
            }
            for special_id in special_ids:
                valid_mask = valid_mask & (input_ids != int(special_id))

            selected_ids = input_ids[valid_mask]
            if selected_ids.numel() == 0:
                raise ValueError("Statement produced no usable text tokens after special-token removal")

            text_token_features = self.model.text_projection(
                text_output.last_hidden_state[0][valid_mask]
            )
            token_labels = tokenizer.convert_ids_to_tokens(selected_ids.detach().cpu().tolist())

            image_global = self.model.visual_projection(vision_output.pooler_output)[0]
            text_global = self.model.text_projection(text_output.pooler_output)[0]
            image_global = F.normalize(image_global, p=2, dim=-1)
            text_global = F.normalize(text_global, p=2, dim=-1)

            visual_patch_features = F.normalize(visual_patch_features, p=2, dim=-1)
            text_token_features = F.normalize(text_token_features, p=2, dim=-1)

            if visual_patch_features.shape[1] != text_token_features.shape[1]:
                raise RuntimeError("CLIP visual and text projection dimensions do not match")

            # ---- TRUST WEIGHTING ----
            # force_full_ot disables partial-OT trust mass entirely.
            # Otherwise, either the legacy "heterogeneity" normalization
            # (sharpness / max(sharpness)) or the proposed "absolute"
            # sigmoid form is used. The sigmoid form is monotone in
            # per-patch sharpness, so monotone image degradation drives
            # discarded mass monotonically upward, which is the intended
            # diagnostic direction. tau and k are dataset constants (fit
            # once on the UCA fit split, then frozen) supplied via env vars.
            if self.force_full_ot:
                visual_trust = torch.ones(
                    visual_patch_features.shape[0], device=self.device, dtype=torch.float32
                )
            elif sharpness_tensor.numel() != visual_patch_features.shape[0]:
                visual_trust = torch.ones(
                    visual_patch_features.shape[0], device=self.device, dtype=torch.float32
                )
            elif self.trust_mode == "absolute":
                visual_trust = torch.sigmoid((sharpness_tensor - self.ot_trust_tau) / self.ot_trust_k).clamp(
                    min=1e-6, max=1.0
                )
            else:  # "heterogeneity"
                if float(sharpness_tensor.max().item()) <= 1e-8:
                    visual_trust = torch.ones_like(sharpness_tensor)
                else:
                    visual_trust = (sharpness_tensor / sharpness_tensor.max()).clamp(
                        min=1e-6, max=1.0
                    )
            text_trust = torch.ones(
                text_token_features.shape[0], device=self.device, dtype=torch.float32
            )

            sinkhorn_divergence, transport_plan, cost_matrix = self.ot_solver(
                visual_patch_features,
                text_token_features,
                visual_trust=visual_trust,
                text_trust=text_trust,
            )
            discarded_mass = float(self.ot_solver.last_diagnostics["discarded_mass"])
            global_similarity = torch.sum(image_global * text_global)
            if not torch.isfinite(image_global).all() or not torch.isfinite(text_global).all():
                raise RuntimeError("CLIP produced non-finite global image/text features")
            if not torch.isfinite(global_similarity):
                raise RuntimeError("CLIP produced a non-finite global similarity")
            discarded_tensor = torch.tensor(
                [discarded_mass], device=image_global.device, dtype=image_global.dtype
            )
            pair_features = torch.cat(
                [
                    torch.abs(image_global - text_global),
                    image_global * text_global,
                    sinkhorn_divergence.reshape(1).to(dtype=image_global.dtype),
                    discarded_tensor,
                ],
                dim=0,
            )

        if not torch.isfinite(pair_features).all():
            raise RuntimeError("CLIP/OT produced non-finite pair features")
        if pair_features.shape[0] != EvidentialPairHead.PAIR_FEATURE_DIM:
            raise RuntimeError(
                f"Unexpected CLIP/OT pair feature width {pair_features.shape[0]}; "
                f"expected {EvidentialPairHead.PAIR_FEATURE_DIM} before quality features"
            )

        return {
            "visual_patch_features": visual_patch_features,
            "text_token_features": text_token_features,
            "token_labels": token_labels,
            "global_similarity": float(global_similarity.detach().cpu().item()),
            "pair_features": pair_features,
            "quality_features": quality_features,
            "visual_quality_metrics": quality_metrics,
            "sinkhorn_divergence": float(sinkhorn_divergence.detach().cpu().item()),
            "standard_entropic_ot_cost": float(self.ot_solver.last_diagnostics["cross_regularized_ot"]),
            "ot_diagnostics": dict(self.ot_solver.last_diagnostics),
            "transport_plan": transport_plan,
            "cost_matrix": cost_matrix,
            "discarded_mass": discarded_mass,
        }

    def extract_pair_features(self, image: Image.Image, statement: str) -> Dict[str, Any]:
        result = self._extract_features(image, statement)
        return {
            "pair_features": result["pair_features"].detach().cpu(),
            "quality_features": result["quality_features"].detach().cpu(),
            "visual_quality_metrics": result["visual_quality_metrics"],
            "global_similarity": result["global_similarity"],
            "sinkhorn_divergence": result["sinkhorn_divergence"],
            "standard_entropic_ot_cost": result["standard_entropic_ot_cost"],
        }

    def analyze_image(self, image: Image.Image, statement: str) -> Dict[str, Any]:
        result = self._extract_features(image, statement)
        quality = result["visual_quality_metrics"]

        assessment_status = "INDETERMINATE"
        indeterminate_reason = "No trained and calibrated evidential classifier is loaded."
        probabilities = None
        epistemic_uncertainty = None
        is_contradiction = None
        quality_in_training_domain = None
        conformal_info: Optional[Dict[str, Any]] = None

        if self.head is not None and self.calibration is not None:
            if self.quality_feature_mean is None or self.quality_feature_std is None:
                raise RuntimeError("Trained quality-feature normalization is not loaded")
            raw_quality_batch = result["quality_features"].unsqueeze(0)
            quality_in_training_domain = self.quality_features_in_training_domain(
                raw_quality_batch,
                self.quality_feature_mean,
                self.quality_feature_std,
            )
            normalized_quality = self.standardize_quality_features(
                raw_quality_batch,
                self.quality_feature_mean,
                self.quality_feature_std,
            )
            classifier_features = torch.cat(
                [result["pair_features"].unsqueeze(0), normalized_quality], dim=1
            )
            with torch.inference_mode():
                alpha, probability_tensor, uncertainty_tensor = self.head(classifier_features)
            if (
                not torch.isfinite(alpha).all()
                or not torch.isfinite(probability_tensor).all()
                or not torch.isfinite(uncertainty_tensor).all()
            ):
                raise RuntimeError("Evidential classifier produced non-finite output; assessment aborted")
            if torch.any(alpha < 1.0) or torch.any(probability_tensor < 0.0) or torch.any(probability_tensor > 1.0):
                raise RuntimeError("Evidential classifier produced invalid Dirichlet probabilities")
            if not torch.allclose(
                probability_tensor.sum(dim=-1),
                torch.ones(probability_tensor.shape[0], device=probability_tensor.device, dtype=probability_tensor.dtype),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError("Evidential class probabilities do not sum to one")
            probabilities = probability_tensor[0].detach().cpu().tolist()
            epistemic_uncertainty = float(uncertainty_tensor[0].detach().cpu().item())
            probability_contradiction = float(probabilities[self.CLASS_TO_INDEX["CONTRADICTION"]])
            max_uncertainty = float(self.calibration["max_epistemic_uncertainty"])
            decision_threshold = float(self.calibration["contradiction_probability_threshold"])

            if not quality_in_training_domain:
                indeterminate_reason = (
                    "At least one image-quality feature lies outside the trained "
                    f"normalization range (more than +/-{self.QUALITY_FEATURE_CLIP:.1f} standard deviations). "
                    "The model abstains rather than extrapolating."
                )
            elif epistemic_uncertainty > max_uncertainty:
                indeterminate_reason = (
                    "The trained evidential head reports uncertainty above the "
                    "empirically calibrated acceptance limit."
                )
            else:
                assessment_status = (
                    "CONTRADICTION" if probability_contradiction >= decision_threshold else "MATCH"
                )
                is_contradiction = assessment_status == "CONTRADICTION"
                indeterminate_reason = None

        if self.conformal is not None:
            from .conformal import extract_conformal_features, prediction_set_to_assessment
            conformal_features = extract_conformal_features({
                "sinkhorn_divergence": result["sinkhorn_divergence"],
                "discarded_mass": result["discarded_mass"],
                "global_clip_similarity": result["global_similarity"],
            })
            prediction_set = self.conformal.predict_set(
                conformal_features.reshape(1, -1)
            )[0]
            assessment_status, indeterminate_reason = prediction_set_to_assessment(
                prediction_set
            )
            is_contradiction = (
                True if assessment_status == "CONTRADICTION"
                else False if assessment_status == "MATCH"
                else None
            )
            conformal_info = {
                "alpha": self.conformal.alpha,
                "q_hat": self.conformal.q_hat,
                "coverage_guarantee": 1.0 - self.conformal.alpha,
                "prediction_set": prediction_set,
            }

        visual_patch_count = int(result["visual_patch_features"].shape[0])
        patches = [f"CLIP patch {index + 1}" for index in range(visual_patch_count)]
        diagnostics = dict(self.ot_solver.last_diagnostics)

        return {
            "assessment_status": assessment_status,
            "indeterminate_reason": indeterminate_reason,
            "is_contradiction": is_contradiction,
            "sinkhorn_divergence": result["sinkhorn_divergence"],
            "standard_entropic_ot_cost": result["standard_entropic_ot_cost"],
            "global_clip_similarity": result["global_similarity"],
            "probabilities": probabilities,
            "model_epistemic_uncertainty": epistemic_uncertainty,
            "visual_quality_metrics": quality,
            "quality_features_used_by_classifier": bool(self.head is not None and self.calibration is not None),
            "quality_features_in_training_domain": quality_in_training_domain,
            "tokens": result["token_labels"],
            "patches": patches,
            "heatmap_data": result["cost_matrix"].detach().cpu().tolist(),
            "transport_plan": result["transport_plan"].detach().cpu().tolist(),
            "ot_diagnostics": diagnostics,
            "discarded_mass": result["discarded_mass"],
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_version": self.calibration.get("schema_version") if self.calibration else None,
            "feature_schema_version": self.FEATURE_SCHEMA_VERSION,
            "conformal": conformal_info,
        }