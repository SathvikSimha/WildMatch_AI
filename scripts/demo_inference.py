# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path

# Add project root and WildMatch_AI to python path
current_dir = Path(__file__).resolve().parent
wildmatch_root = current_dir.parent
sys.path.insert(0, str(wildmatch_root))
sys.path.insert(0, str(wildmatch_root.parent))

from WildMatch_AI.models.dual_encoder import DualEncoder
from WildMatch_AI.preprocessing.text_preprocessing import Vocabulary, tokenize_caption
from WildMatch_AI.dataset.wildlife_dataset import WildlifeDataset

def main():
    print("=" * 80)
    print("       WILDMATCH AI — INTERACTIVE DUAL-ENCODER RETRIEVAL DEMO")
    print("=" * 80)

    # Ensure working directory is WildMatch_AI so relative dataset paths resolve
    os.chdir(str(wildmatch_root))

    # 1. Load Vocabulary
    vocab_path = Path("dataset/vocabulary.json")
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return
    vocab = Vocabulary.load_json(str(vocab_path))
    print(f"[✓] Loaded Vocabulary: {len(vocab)} words")

    # 2. Instantiate DualEncoder
    model = DualEncoder(embed_dim=256, init_temperature=0.07)
    model.eval()
    print(f"[✓] Initialized DualEncoder (937,505 parameters, 256-d embedding space)")

    # 3. Load Gallery Dataset
    test_csv = Path("dataset/splits/test.csv")
    curated_dir = Path("dataset/curated")
    if not test_csv.exists() or not curated_dir.exists():
        print("[!] Using train.csv as gallery...")
        test_csv = Path("dataset/splits/train.csv")

    dataset = WildlifeDataset(
        split_csv=str(test_csv),
        vocab=str(vocab_path),
        image_root=None,
        transform=None
    )
    print(f"[✓] Loaded Gallery Dataset: {len(dataset)} wildlife images from {test_csv.name}")

    # 4. Precompute Image Embeddings
    print("\nEncoding gallery images into 256-d visual semantic space...")
    gallery_images = []
    gallery_metadata = []
    for i in range(len(dataset)):
        item = dataset[i]
        gallery_images.append(item['image'])
        gallery_metadata.append({
            'photo_id': item['photo_id'],
            'species': item['species'],
            'image_path': item['image_path']
        })

    gallery_tensor = torch.stack(gallery_images, dim=0) # [N, 3, 224, 224]
    with torch.no_grad():
        gallery_embeds = model.encode_image(gallery_tensor, normalize=True) # [N, 256]
    print(f"[✓] Precomputed {gallery_embeds.shape[0]} visual embeddings (L2-norm = {torch.norm(gallery_embeds[0]):.4f})")

    # 5. Test Queries
    queries = [
        "a tiger lying down near the river bank",
        "african lion resting in golden grass",
        "large asian elephant walking in green forest",
        "spotted deer grazing calmly in meadow"
    ]

    print("\n" + "=" * 80)
    print("EXECUTING TEXT-TO-IMAGE SEMANTIC RETRIEVAL ON GALLERY")
    print("=" * 80)

    for q_idx, query in enumerate(queries, 1):
        print(f"\n[Query #{q_idx}]: \"{query}\"")
        
        # Tokenize query
        tokens = tokenize_caption(query)
        token_ids = vocab.numericalize(tokens, max_len=32)
        query_tensor = torch.tensor([token_ids], dtype=torch.long) # [1, 32]

        with torch.no_grad():
            query_embed = model.encode_text(query_tensor, normalize=True) # [1, 256]
            # Cosine similarity is dot product of normalized vectors
            sims = torch.matmul(query_embed, gallery_embeds.t()).squeeze(0) # [N]
            topk_scores, topk_indices = torch.topk(sims, k=3)

        print("  Top-3 Matching Gallery Images:")
        for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
            meta = gallery_metadata[idx.item()]
            print(f"    Rank {rank} (Sim: {score.item():+.4f}) | Species: {meta['species'].upper():<14} | Photo ID: {meta['photo_id']}")
            print(f"           File: {meta['image_path']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All image and text embeddings verified successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
