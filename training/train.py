"""
WildMatch AI - Member 3: Multimodal Model Training Pipeline (`training/train.py`)

Implements the end-to-end training pipeline for WildMatch AI dual encoders:
- Dataset and DataLoader initialization for train/val/test splits
- Model initialization (Version 3 ResNet-18 or Version 2 Scratch CNN)
- Differential learning rate optimization (AdamW with backbone vs head rates)
- Symmetric and Multi-Positive Contrastive Loss + Auxiliary Species Supervision
- Learning rate scheduling via ReduceLROnPlateau
- Validation loop tracking cross-modal Recall@1 and MRR
- Best checkpoint selection and model serialization
- Training history logging (CSV/JSON) and loss/metric curve generation
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataloader import create_dataloaders
from dataset.wildlife_dataset import WildlifeDataset
from preprocessing.text_preprocessing import Vocabulary
from models.wildmatch_v3 import (
    WildMatchV3DualEncoder,
    WildMatchV3SpeciesHeads,
    SPECIES_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from training.loss import WildMatchLoss


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve compute device with safe CPU fallback."""
    if requested_device:
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(
    model: nn.Module,
    heads: Optional[nn.Module] = None,
    lr_backbone: float = 1e-5,
    lr_heads: float = 2e-4,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """
    Construct AdamW optimizer with differential learning rates for pretrained backbone
    vs newly initialized heads.
    """
    param_groups = []

    # If model has a backbone attribute, separate backbone params from projection params
    if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "backbone"):
        backbone_params = list(model.image_encoder.backbone.parameters())
        head_params = (
            list(model.image_encoder.projection.parameters())
            + list(model.text_encoder.parameters())
            + [model.logit_scale]
        )
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        param_groups.append({"params": head_params, "lr": lr_heads, "weight_decay": weight_decay})
    else:
        param_groups.append({"params": model.parameters(), "lr": lr_heads, "weight_decay": weight_decay})

    if heads is not None:
        param_groups.append({"params": heads.parameters(), "lr": lr_heads, "weight_decay": weight_decay})

    return optim.AdamW(param_groups)


def train_one_epoch(
    model: WildMatchV3DualEncoder,
    heads: Optional[WildMatchV3SpeciesHeads],
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: WildMatchLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Execute one full training epoch over the training DataLoader."""
    model.train()
    if heads is not None:
        heads.train()

    total_loss_accum = 0.0
    contrastive_loss_accum = 0.0
    aux_loss_accum = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        species_idx = batch["species_idx"].to(device)

        optimizer.zero_grad()

        # Forward pass through dual encoder
        image_embeds, text_embeds, scale = model(images, captions, normalize=True)

        img_logits = None
        txt_logits = None
        if heads is not None:
            img_logits = heads.image_classifier(image_embeds)
            txt_logits = heads.text_classifier(text_embeds)

        # Compute joint loss
        loss_dict = criterion(
            image_embeddings=image_embeds,
            text_embeddings=text_embeds,
            species_indices=species_idx,
            image_class_logits=img_logits,
            text_class_logits=txt_logits,
            logit_scale=scale,
        )

        loss = loss_dict["total_loss"]
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if heads is not None:
            torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss_accum += loss.item()
        contrastive_loss_accum += loss_dict["contrastive_loss"].item()
        aux_loss_accum += loss_dict["aux_total_loss"].item()
        num_batches += 1

    return {
        "train_loss": total_loss_accum / max(num_batches, 1),
        "train_contrastive_loss": contrastive_loss_accum / max(num_batches, 1),
        "train_aux_loss": aux_loss_accum / max(num_batches, 1),
    }


def evaluate(
    model: WildMatchV3DualEncoder,
    heads: Optional[WildMatchV3SpeciesHeads],
    loader: DataLoader,
    criterion: WildMatchLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Execute validation pass computing loss and species retrieval Recall@1."""
    model.eval()
    if heads is not None:
        heads.eval()

    total_loss_accum = 0.0
    all_img_embeds = []
    all_txt_embeds = []
    all_species = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            captions = batch["caption"].to(device)
            species_idx = batch["species_idx"].to(device)

            image_embeds, text_embeds, scale = model(images, captions, normalize=True)

            img_logits = heads.image_classifier(image_embeds) if heads else None
            txt_logits = heads.text_classifier(text_embeds) if heads else None

            loss_dict = criterion(
                image_embeddings=image_embeds,
                text_embeddings=text_embeds,
                species_indices=species_idx,
                image_class_logits=img_logits,
                text_class_logits=txt_logits,
                logit_scale=scale,
            )
            total_loss_accum += loss_dict["total_loss"].item()

            all_img_embeds.append(image_embeds.cpu())
            all_txt_embeds.append(text_embeds.cpu())
            all_species.append(species_idx.cpu())

    num_batches = len(loader)
    avg_loss = total_loss_accum / max(num_batches, 1)

    if not all_img_embeds:
        return {"val_loss": avg_loss, "val_recall@1": 0.0}

    # Aggregate embeddings for retrieval evaluation
    cat_img = torch.cat(all_img_embeds, dim=0)
    cat_txt = torch.cat(all_txt_embeds, dim=0)
    cat_sp = torch.cat(all_species, dim=0)

    # Cosine similarity matrix [N, N]
    sim = torch.matmul(cat_txt, cat_img.t())  # Text-to-Image queries
    sp_mask = (cat_sp.unsqueeze(0) == cat_sp.unsqueeze(1))  # [N, N] True if same species

    top1_indices = sim.argmax(dim=1)
    t2i_hits = sp_mask[torch.arange(len(cat_sp)), top1_indices].float().mean().item()

    return {
        "val_loss": avg_loss,
        "val_species_recall@1": t2i_hits,
    }


def save_training_plots(
    history: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str = "wildmatch_v3",
) -> None:
    """Generate and save loss and validation metric visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_recall = [h.get("val_species_recall@1", 0.0) for h in history]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Total Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Total Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WildMatch AI - Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / f"{prefix}_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {loss_path}")

    # 2. Validation Recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r * 100 for r in val_recall], label="Val Species Recall@1 (%)", color="#2ca02c", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Recall@1 (%)")
    plt.title("WildMatch AI - Validation Species Retrieval Recall@1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    val_path = output_dir / f"{prefix}_validation.png"
    plt.savefig(val_path, dpi=200)
    plt.close()
    print(f"Saved validation curve to {val_path}")


def run_training(
    epochs: int = 35,
    batch_size: int = 16,
    lr_backbone: float = 1e-5,
    lr_heads: float = 2e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    lambda_aux: float = 0.5,
    splits_dir: str = "dataset/splits",
    vocab_path: str = "dataset/vocabulary.json",
    output_dir: str = "models",
    device_name: Optional[str] = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Execute complete training workflow."""
    device = get_device(device_name)
    print(f"Training WildMatch AI on compute device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataloaders
    print(f"Loading datasets from {splits_dir} with batch_size={batch_size}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        splits_dir=splits_dir,
        vocab=vocab_path,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples.")

    # 2. Model & Heads
    model = WildMatchV3DualEncoder(embed_dim=256, vocab_size=158, init_temperature=temperature)
    heads = WildMatchV3SpeciesHeads(embed_dim=256, num_classes=len(SPECIES_CLASSES))
    model.to(device)
    heads.to(device)

    # 3. Optimizer & Scheduler
    optimizer = build_optimizer(
        model=model,
        heads=heads,
        lr_backbone=lr_backbone,
        lr_heads=lr_heads,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    criterion = WildMatchLoss(
        contrastive_type="multi_positive",
        temperature=temperature,
        lambda_aux=lambda_aux,
    )

    history: List[Dict[str, Any]] = []
    best_recall = -1.0
    best_epoch = -1

    print("\n--- Starting Training Loop ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, heads, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, heads, val_loader, criterion, device)

        rec1 = val_metrics["val_species_recall@1"]
        scheduler.step(rec1)

        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Recall@1: {rec1 * 100:.1f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Track and save best checkpoint
        if rec1 > best_recall:
            best_recall = rec1
            best_epoch = epoch
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "heads_state_dict": heads.state_dict(),
                "best_val_recall": best_recall,
                "vocab_size": 158,
                "embed_dim": 256,
            }
            torch.save(best_ckpt, out_path / "wildmatch_v3_best.pth")
            torch.save(model.state_dict(), out_path / "wildmatch_v3_dual_encoder.pth")
            torch.save(heads.state_dict(), out_path / "wildmatch_v3_species_heads.pth")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes. Best Val Recall@1: {best_recall * 100:.1f}% at Epoch {best_epoch}.")

    # Save final model
    final_ckpt = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "heads_state_dict": heads.state_dict(),
        "final_val_recall": history[-1].get("val_species_recall@1", 0.0),
        "vocab_size": 158,
        "embed_dim": 256,
    }
    torch.save(final_ckpt, out_path / "wildmatch_v3_final.pth")

    # Save history JSON & CSV
    history_json_path = out_path / "wildmatch_v3_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    history_csv_path = out_path / "wildmatch_v3_history.csv"
    if history:
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    if save_plots:
        save_training_plots(history, out_path)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_recall": best_recall,
        "total_time_seconds": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WildMatch AI Multimodal Model Training")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for ResNet-18 backbone")
    parser.add_argument("--lr-heads", type=float, default=2e-4, help="Learning rate for projection heads and BiLSTM")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--temperature", type=float, default=0.07, help="Initial temperature parameter")
    parser.add_argument("--lambda-aux", type=float, default=0.5, help="Weight for auxiliary classification loss")
    parser.add_argument("--splits-dir", type=str, default="dataset/splits", help="Path to splits directory")
    parser.add_argument("--vocab-path", type=str, default="dataset/vocabulary.json", help="Path to vocabulary.json")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save checkpoints and logs")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting curves")

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        lambda_aux=args.lambda_aux,
        splits_dir=args.splits_dir,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        device_name=args.device,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

