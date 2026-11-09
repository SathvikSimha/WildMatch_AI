"""
WildMatch AI - Version 3 Sample Inference Script.

Demonstrates how Member 4 can load the trained dual-encoder and auxiliary
species classification heads, preprocess multimodal inputs, generate embeddings,
and perform semantic similarity matching and species prediction.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference import (
    load_model,
    load_species_heads,
    preprocess_image,
    tokenize_caption,
    embed_image,
    embed_text,
    compute_similarity,
)
from models.wildmatch_v3 import SPECIES_CLASSES


def main() -> None:
    print("=" * 80)
    print("      WILDMATCH AI (VERSION 3) — PRODUCTION INFERENCE DEMONSTRATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Device: {device.upper()}")

    # 1. Load Trained Checkpoint
    checkpoint_path = PROJECT_ROOT / "models" / "wildmatch_v3_dual_encoder.pth"
    print(f"\n[1/5] Loading dual encoder from {checkpoint_path.name}...")
    model = load_model(checkpoint_path, device=device)
    print(f"      [✓] Loaded WildMatchV3DualEncoder (ResNet-18 + BiLSTM)")
    print(f"      [✓] Logit scale parameter: {model.logit_scale.item():.4f}")

    # 2. Load Auxiliary Heads
    heads_path = PROJECT_ROOT / "models" / "wildmatch_v3_species_heads.pth"
    print(f"\n[2/5] Loading auxiliary classification heads from {heads_path.name}...")
    heads = load_species_heads(heads_path, device=device)
    print(f"      [✓] Loaded Species Heads: {heads.species}")

    # 3. Text Queries
    queries = [
        "a majestic tiger resting beside water in the jungle",
        "african lion lying in tall golden grass in open field",
        "herd of asian elephants grazing peacefully on green hillside",
        "spotted chital deer standing alert near forest trees",
    ]

    print(f"\n[3/5] Tokenizing and embedding {len(queries)} text queries...")
    text_embeddings = []
    for query in queries:
        tokens = tokenize_caption(query, device=device)
        emb = embed_text(model, tokens)
        norm = torch.norm(emb, p=2, dim=-1).item()
        _, labels, probs = heads.predict_text(emb)
        top_prob = probs.max().item() * 100.0
        print(f"      Query: \"{query}\"")
        print(f"             Shape: {list(emb.shape)}, L2 Norm: {norm:.4f}")
        print(f"             Predicted Species: {labels[0]} ({top_prob:.1f}% confidence)")
        text_embeddings.append(emb)

    text_tensor = torch.cat(text_embeddings, dim=0)  # [4, 256]

    # 4. Image Preprocessing & Embedding
    print(f"\n[4/5] Simulating standardized wildlife image batches...")
    # Generate representative synthetic images to verify inference pipeline
    synthetic_images = torch.rand(4, 3, 224, 224, device=device)
    # Apply standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalized_images = (synthetic_images - mean) / std

    image_tensor = embed_image(model, normalized_images)
    print(f"      [✓] Encoded image batch to shape: {list(image_tensor.shape)}")
    img_norms = torch.norm(image_tensor, p=2, dim=-1)
    print(f"      [✓] Image L2 Norms min/max: {img_norms.min().item():.4f} / {img_norms.max().item():.4f}")

    _, img_labels, img_probs = heads.predict_image(image_tensor)
    for i in range(len(img_labels)):
        print(f"      Image #{i + 1} classified as: {img_labels[i]} (confidence: {img_probs[i].max().item() * 100:.1f}%)")

    # 5. Multimodal Similarity Evaluation
    print(f"\n[5/5] Computing Cross-Modal Cosine Similarity Matrix...")
    similarity_matrix = compute_similarity(image_tensor, text_tensor)
    print(f"      Similarity Matrix Shape: {list(similarity_matrix.shape)} (Images x Queries)")
    print(f"      Similarity Matrix:\n{similarity_matrix.cpu().numpy()}")

    print("\n" + "=" * 80)
    print("      INFERENCE PIPELINE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

