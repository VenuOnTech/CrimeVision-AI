"""Train the supervised evidential pair classifier from human-labelled data.

CSV schema:
image_path,statement,label,camera_id,environment_id,source_recording_id
label must be MATCH or CONTRADICTION. Camera IDs, environment IDs and original
source_recording_id values must not overlap between the train and validation
splits. The script never creates synthetic labels or images.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data_utils import (
    assert_camera_disjoint,
    assert_environment_disjoint,
    assert_source_recording_disjoint,
    label_to_index,
    load_annotation_csv,
    open_rgb_image,
)
from src.evidential_fusion import EvidentialPairHead, evidential_classification_loss
from src.inference import EvidentialInferenceEngine


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_features(
    rows: List[Dict[str, str | Path]],
    engine: EvidentialInferenceEngine,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    quality_features = []
    labels = []
    for index, row in enumerate(rows, start=1):
        image = open_rgb_image(Path(row["image_path"]))
        item = engine.extract_pair_features(image, str(row["statement"]))
        features.append(item["pair_features"].float())
        quality_features.append(item["quality_features"].float())
        labels.append(label_to_index(str(row["label"])))
        if index % 10 == 0 or index == len(rows):
            print(f"[{split_name}] extracted {index}/{len(rows)} examples")
    return torch.stack(features), torch.stack(quality_features), torch.tensor(labels, dtype=torch.long)


def balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels_np = labels.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()
    recalls = []
    for class_index in (0, 1):
        mask = labels_np == class_index
        if not np.any(mask):
            raise ValueError("Both MATCH and CONTRADICTION classes are required in each split")
        recalls.append(float(np.mean(predictions_np[mask] == class_index)))
    return float(np.mean(recalls))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_csv = Path(args.train_csv).expanduser().resolve()
    validation_csv = Path(args.validation_csv).expanduser().resolve()
    train_rows = load_annotation_csv(train_csv)
    validation_rows = load_annotation_csv(validation_csv)
    named_splits = {"train": train_rows, "validation": validation_rows}
    assert_camera_disjoint(named_splits)
    assert_environment_disjoint(named_splits)
    assert_source_recording_disjoint(named_splits)

    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        labels = {str(row["label"]) for row in rows}
        if labels != {"MATCH", "CONTRADICTION"}:
            raise ValueError(f"{name} split must contain both MATCH and CONTRADICTION examples")

    device = args.device
    engine = EvidentialInferenceEngine(model_name=args.model_name, model_revision=args.model_revision, device=device)
    print(f"[model] {engine.model_name}@{engine.model_revision}")
    pair_train, quality_train, y_train = prepare_features(train_rows, engine, "train")
    pair_val, quality_val, y_val = prepare_features(validation_rows, engine, "validation")

    # Quality metrics have different physical scales. Fit normalization ONLY
    # on the training split, then reuse these exact values for validation,
    # calibration, testing and deployed inference.
    quality_mean = quality_train.mean(dim=0)
    quality_std = quality_train.std(dim=0, unbiased=False)
    quality_std = torch.where(quality_std > 1e-8, quality_std, torch.ones_like(quality_std))
    quality_train_z = engine.standardize_quality_features(quality_train, quality_mean, quality_std)
    quality_val_z = engine.standardize_quality_features(quality_val, quality_mean, quality_std)

    x_train = torch.cat([pair_train, quality_train_z], dim=1).to(engine.device)
    x_val = torch.cat([pair_val, quality_val_z], dim=1).to(engine.device)
    quality_mean = quality_mean.detach().cpu()
    quality_std = quality_std.detach().cpu()
    y_train = y_train.to(engine.device)
    y_val = y_val.to(engine.device)

    model = EvidentialPairHead().to(engine.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # WeightedRandomSampler operates with a CPU generator. Keep its weights on
    # CPU even when the model and feature tensors are placed on CUDA.
    labels_cpu = y_train.detach().cpu()
    class_counts = torch.bincount(labels_cpu, minlength=EvidentialPairHead.NUM_CLASSES).float()
    if torch.any(class_counts == 0):
        raise ValueError("Training split must contain examples from both classes")
    per_class_weight = 1.0 / class_counts
    sample_weights = per_class_weight[labels_cpu]
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
    )

    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    best_state = None
    patience_remaining = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        annealing = min(1.0, epoch / max(1, args.kl_anneal_epochs))
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            alpha, _, _ = model(batch_features)
            loss = evidential_classification_loss(alpha, batch_labels, annealing=args.kl_weight * annealing)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; aborting rather than saving corrupt weights")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient norm; aborting rather than saving corrupt weights")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError("Optimizer produced non-finite model parameters; training aborted")
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.inference_mode():
            val_alpha, val_probabilities, val_uncertainty = model(x_val)
            val_loss = evidential_classification_loss(
                val_alpha, y_val, annealing=args.kl_weight
            )
            if (
                not torch.isfinite(val_alpha).all()
                or not torch.isfinite(val_probabilities).all()
                or not torch.isfinite(val_uncertainty).all()
                or not torch.isfinite(val_loss)
            ):
                raise RuntimeError("Non-finite validation output; refusing to save this training run")
            val_predictions = val_probabilities.argmax(dim=1)
            val_bal_acc = balanced_accuracy(y_val, val_predictions)

        print(
            f"Epoch {epoch:03d}: train_loss={np.mean(epoch_losses):.5f}, "
            f"val_loss={float(val_loss.item()):.5f}, val_balanced_accuracy={val_bal_acc:.4f}, "
            f"mean_val_uncertainty={float(val_uncertainty.mean().item()):.4f}"
        )

        improved = (val_bal_acc > best_balanced_accuracy) or (
            np.isclose(val_bal_acc, best_balanced_accuracy) and float(val_loss.item()) < best_validation_loss
        )
        if improved:
            best_balanced_accuracy = val_bal_acc
            best_validation_loss = float(val_loss.item())
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_remaining = args.patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                print("Early stopping: validation metric did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model_name": args.model_name,
        "model_revision": engine.model_revision,
        "input_dim": EvidentialPairHead.INPUT_DIM,
        "class_to_index": {"MATCH": 0, "CONTRADICTION": 1},
        "feature_schema_version": EvidentialInferenceEngine.FEATURE_SCHEMA_VERSION,
        "train_camera_ids": sorted({str(row["camera_id"]) for row in train_rows}),
        "validation_camera_ids": sorted({str(row["camera_id"]) for row in validation_rows}),
        "train_environment_ids": sorted({str(row["environment_id"]) for row in train_rows}),
        "validation_environment_ids": sorted({str(row["environment_id"]) for row in validation_rows}),
        "train_source_recording_ids": sorted({str(row["source_recording_id"]) for row in train_rows}),
        "validation_source_recording_ids": sorted({str(row["source_recording_id"]) for row in validation_rows}),
        "quality_feature_names": list(EvidentialInferenceEngine.QUALITY_FEATURE_NAMES),
        "quality_feature_mean": quality_mean.tolist(),
        "quality_feature_std": quality_std.tolist(),
        "quality_feature_clip_abs_z": EvidentialInferenceEngine.QUALITY_FEATURE_CLIP,
        "train_csv_sha256": sha256_file(train_csv),
        "validation_csv_sha256": sha256_file(validation_csv),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved trained evidential head: {output_path}")
    print(f"Checkpoint SHA-256: {sha256_file(output_path)}")
    print("Next: calibrate this exact checkpoint on an independent camera-disjoint calibration CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--output", default="models/evidential_pair_head.pt")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default=None, help="Immutable Hugging Face revision/commit; defaults to CLIP_MODEL_REVISION or main")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--kl-anneal-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.kl_weight < 0:
        parser.error("learning-rate must be positive; weight-decay and kl-weight cannot be negative")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


if __name__ == "__main__":
    train(parse_args())
