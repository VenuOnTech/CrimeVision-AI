from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.conformal import fit_conformal
from src.evidential_fusion import EvidentialOptimalTransport
from src.inference import EvidentialInferenceEngine


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def convert_ids_to_tokens(self, ids):
        return [f"token_{token_id}" for token_id in ids]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def __call__(self, **kwargs):
        return {
            "pixel_values": torch.zeros(1, 3, 32, 32),
            "input_ids": torch.tensor([[1, 11, 12, 13, 2, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long),
        }


class FakeVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_layernorm = nn.LayerNorm(512)

    def forward(self, pixel_values, return_dict=True):
        hidden = torch.arange(5 * 512, dtype=torch.float32).reshape(1, 5, 512) / 1000.0
        pooled = self.post_layernorm(hidden[:, 0, :])
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(512)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        hidden = torch.arange(6 * 512, dtype=torch.float32).reshape(1, 6, 512) / 1500.0
        hidden = self.final_layer_norm(hidden)
        pooled = hidden[:, 4, :]
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)


class FakeCLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = FakeVisionEncoder()
        self.text_model = FakeTextEncoder()
        self.visual_projection = nn.Linear(512, 512, bias=False)
        self.text_projection = nn.Linear(512, 512, bias=False)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )


def make_engine():
    engine = EvidentialInferenceEngine.__new__(EvidentialInferenceEngine)
    engine.device = torch.device("cpu")
    engine.model_name = "fake-clip"
    engine.model_revision = "fake-revision"
    engine.processor = FakeProcessor()
    engine.model = FakeCLIPModel()
    engine.ot_solver = EvidentialOptimalTransport(
        epsilon=0.5,
        max_iterations=1000,
        tolerance=1e-7,
    )
    engine.head = None
    engine.calibration = None
    engine.conformal = None
    engine.checkpoint_sha256 = None
    engine.quality_feature_mean = None
    engine.quality_feature_std = None
    engine.force_full_ot = False
    return engine


def test_extracts_real_sequence_shapes_and_returns_diagnostic_matrix():
    engine = make_engine()
    image = Image.new("RGB", (32, 32), color=(50, 60, 70))
    result = engine._extract_features(image, "The suspect wore a dark jacket")

    assert result["visual_patch_features"].shape == (4, 512)
    assert result["text_token_features"].shape == (3, 512)
    assert result["token_labels"] == ["token_11", "token_12", "token_13"]
    assert result["cost_matrix"].shape == (4, 3)
    assert result["transport_plan"].shape == (4, 3)
    assert result["pair_features"].shape == (1026,)
    assert result["discarded_mass"] == 0.0
    assert result["quality_features"].shape == (3,)
    assert torch.isfinite(result["pair_features"]).all()
    assert torch.isfinite(result["cost_matrix"]).all()


def test_untrained_engine_is_indeterminate_and_never_infers_a_match():
    engine = make_engine()
    image = Image.new("RGB", (32, 32), color=(50, 60, 70))
    result = engine.analyze_image(image, "There is a vehicle in the image")

    assert result["assessment_status"] == "INDETERMINATE"
    assert result["is_contradiction"] is None
    assert result["probabilities"] is None
    assert result["model_epistemic_uncertainty"] is None
    assert result["quality_features_used_by_classifier"] is False
    assert result["conformal"] is None


def test_engine_uses_conformal_prediction_when_configured():
    import numpy as np
    from src.conformal import fit_conformal

    # Create two SEPARATE datasets (fit and calibration)
    rng = np.random.default_rng(0)
    
    # Fit set
    fit_match = rng.normal(size=(40, 3)) * 0.05 + np.array([0.1, 0.1, 0.5])
    fit_contra = rng.normal(size=(40, 3)) * 0.05 + np.array([0.9, 0.1, 0.9])
    fit_features = np.vstack([fit_match, fit_contra])
    fit_labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)

    # Calibration set (different random seed)
    rng2 = np.random.default_rng(1)
    cal_match = rng2.normal(size=(40, 3)) * 0.05 + np.array([0.1, 0.1, 0.5])
    cal_contra = rng2.normal(size=(40, 3)) * 0.05 + np.array([0.9, 0.1, 0.9])
    cal_features = np.vstack([cal_match, cal_contra])
    cal_labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)

    # Pass all 5 arguments
    artifact = fit_conformal(
        fit_features, fit_labels, 
        cal_features, cal_labels, 
        alpha=0.1
    )

    engine = make_engine()
    engine.conformal = artifact

    image = Image.new("RGB", (32, 32), color=(50, 60, 70))
    result = engine.analyze_image(image, "The suspect wore a dark jacket")

    assert "conformal" in result
    conformal = result["conformal"]
    assert conformal is not None
    assert 0.0 < conformal["alpha"] < 1.0
    assert isinstance(conformal["prediction_set"], list)
    assert all(isinstance(v, int) for v in conformal["prediction_set"])
    assert result["assessment_status"] in {"MATCH", "CONTRADICTION", "INDETERMINATE"}