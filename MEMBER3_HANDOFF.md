# WildMatch AI — Member 3 Model Training & Inference Handoff (`MEMBER3_HANDOFF.md`)

**Target Audience:** Member 4 (Retrieval, Similarity Matching & Gradio UI Developer), Project Evaluators  
**Author:** WildMatch AI — Member 3 (Contrastive Learning & Model Training Specialist)  
**Date:** November 2026  
**Status:** Version 3 Training Completed, Deliverables Verified, Inference API Ready  

---

## 1. Executive Summary

This document establishes the official downstream integration handoff from **Member 3 (Model Training)** to **Member 4 (Retrieval, Evaluation & UI Application)**.

Member 3 has completed the **Version 3** training experiment for **WildMatch AI: Wildlife Image-Text Semantic Matching and Retrieval**. All trained model weights, auxiliary classification heads, vocabulary files, loss history, confusion matrices, and metrics have been finalized, packaged, and verified.

> [!IMPORTANT]
> **Key Integration Guarantee for Member 4:**  
> Member 4 does **NOT** need to run training, execute Jupyter notebooks, or download external weights. All inference components can be loaded directly from the provided checkpoints using lightweight, self-contained PyTorch code (`models.inference` or `models.wildmatch_v3`).

---

## 2. Final Model Architecture

The Version 3 dual-encoder architecture unifies a deep convolutional visual backbone with a recurrent natural language encoder mapped into a shared 256-dimensional unit hypersphere $\mathbb{S}^{255}$:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                 WILDMATCH AI VERSION 3                  │
                              │           DUAL-ENCODER RETRIEVAL PIPELINE               │
                              └─────────────────────────────────────────────────────────┘

        [Wildlife Image]                                                        [Query Text / Caption]
         [B, 3, 224, 224]                                                            [B, 32]
                │                                                                       │
                ▼                                                                       ▼
    ┌───────────────────────┐                                               ┌───────────────────────┐
    │   ResNet-18 Backbone  │ (ImageNet-pretrained,                         │    Word Embedding     │ (158 vocab -> 128-d,
    │   (Conv1..Layer4)     │  partially fine-tuned)                        │    + Dropout (p=0.1)  │  <PAD>=0 zeroed)
    └───────────┬───────────┘                                               └───────────┬───────────┘
                │ [B, 512, 7, 7]                                                        │ [B, 32, 128]
                ▼                                                                       ▼
    ┌───────────────────────┐                                               ┌───────────────────────┐
    │ AdaptiveAvgPool2d(1,1)│                                               │  Bidirectional LSTM   │ (hidden_dim=128/dir
    │      + Flatten        │                                               │       (1 Layer)       │  -> 256-d output)
    └───────────┬───────────┘                                               └───────────┬───────────┘
                │ [B, 512]                                                              │ [B, 32, 256]
                ▼                                                                       ▼
    ┌───────────────────────┐                                               ┌───────────────────────┐
    │ Visual Projection Head│ Linear(512, 512) -> ReLU                      │Text Projection Head   │ Masked Mean Pooling
    │  (512 -> 512 -> 256)  │ -> Dropout(0.1) -> Linear(512, 256)           │  (256 -> 256 -> 256)  │ -> Linear(256, 256) -> ReLU
    └───────────┬───────────┘                                               └───────────┬───────────┘ -> Dropout -> Linear(256, 256)
                │ [B, 256]                                                              │ [B, 256]
                ▼                                                                       ▼
    ┌───────────────────────┐                                               ┌───────────────────────┐
    │ Unit L2 Normalization │ ||v||_2 = 1.0                                 │ Unit L2 Normalization │ ||t||_2 = 1.0
    └───────────┬───────────┘                                               └───────────┬───────────┘
                │                                                                       │
                ├───────────────────────────────────┬───────────────────────────────────┤
                │                                   │                                   │
                ▼                                   ▼                                   ▼
    ┌───────────────────────┐           ┌───────────────────────┐           ┌───────────────────────┐
    │  Auxiliary Image Head │           │ Cross-Modal Matching  │           │  Auxiliary Text Head  │
    │   Linear(256 -> 4)    │           │ S = (v · t^T)         │           │   Linear(256 -> 4)    │
    │   Species Logits      │           │ Scaled: logits = S/τ  │           │   Species Logits      │
    └───────────────────────┘           └───────────────────────┘           └───────────────────────┘
```

### Architectural Specifications Table

| Component | Technical Specification | Dimensions / Parameters |
|---|---|---|
| **Visual Backbone** | ImageNet-pretrained ResNet-18 (Conv1 to Layer4) | $[B, 3, 224, 224] \to [B, 512, 7, 7]$ |
| **Spatial Pooling** | `AdaptiveAvgPool2d((1, 1))` + `Flatten()` | $[B, 512, 7, 7] \to [B, 512]$ |
| **Image Projection Head** | `Linear(512, 512)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(512, 256)` | $[B, 512] \to [B, 256]$ |
| **Image Normalization** | $L_2$ unit normalization: $\mathbf{v} = \mathbf{z}_v / \|\mathbf{z}_v\|_2$ | $\|\mathbf{v}\|_2 = 1.0 \in \mathbb{R}^{256}$ |
| **Word Embedding** | Dense embedding with `<PAD>=0` index masking | $V = 158 \to D_{word} = 128$ |
| **Recurrent Backbone** | 1-layer Bidirectional LSTM (`batch_first=True`) | $128 \to 2 \times 128 = 256$ channels |
| **Temporal Pooling** | Padding-aware masked mean pooling across non-PAD tokens | $[B, 32, 256] \to [B, 256]$ |
| **Text Projection Head** | `Linear(256, 256)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(256, 256)` | $[B, 256] \to [B, 256]$ |
| **Text Normalization** | $L_2$ unit normalization: $\mathbf{t} = \mathbf{z}_t / \|\mathbf{z}_t\|_2$ | $\|\mathbf{t}\|_2 = 1.0 \in \mathbb{R}^{256}$ |
| **Shared Embedding Dim** | Joint multimodal latent space | $D = 256$ |
| **Learnable Scale ($\tau$)** | `logit_scale` parameter initialized to $\ln(1/0.07)$ | $\text{scale} \approx 14.2857$ |
| **Auxiliary Heads** | Separate single-layer classifiers for image and text | `Linear(256, 4)` |

---

## 3. Checkpoint Deliverables & File Manifest

All files exist in both `models/` and the project root directory:

| Filename | File Size | Description & Downstream Purpose |
|---|---|---|
| `models/wildmatch_v3_dual_encoder.pth` | ~48.0 MB | Full trained state dict of `WildMatchV3DualEncoder` (ResNet-18, BiLSTM, projections, `logit_scale`). Primary model checkpoint for Member 4. |
| `models/wildmatch_v3_species_heads.pth` | ~11.1 KB | Trained weights of auxiliary image and text classification heads (`image_classifier`, `text_classifier`). Used for species tagging and zero-shot confidence scoring. |
| `models/wildmatch_v3_best.pth` | ~48.0 MB | Complete training checkpoint at Epoch 19 including optimizer state, species list, history, and metadata. |
| `models/wildmatch_v3_vocabulary.json` | ~9.2 KB | Serialized vocabulary JSON containing 158 words, token frequencies, special token IDs (`<PAD>:0`, `<UNK>:1`, `<BOS>:2`, `<EOS>:3`). |
| `models/wildmatch_v3_metrics.json` | ~700 B | Official experimental metrics evaluated on the held-out test set of 30 samples. |
| `models/wildmatch_v3_history.json` | ~7.6 KB | Per-epoch training losses, validation species retrieval, and learning rates across 34 epochs. |
| `models/wildmatch_v3_history.csv` | ~5.6 KB | CSV table of training dynamics for reporting and plotting in Member 4 UI. |
| `models/wildmatch_v3_loss.png` | ~60.8 KB | Multi-panel plot illustrating total loss, contrastive loss, image CE loss, and text CE loss over epochs. |
| `models/wildmatch_v3_validation.png` | ~96.1 KB | Validation tracking plot showing species retrieval Recall@1 reaching 100.0%. |
| `models/wildmatch_v3_image_confusion.png`| ~60.5 KB | 4x4 Confusion matrix of image species classifier on the held-out test set. |
| `models/wildmatch_v3_text_confusion.png` | ~61.2 KB | 4x4 Confusion matrix of text species classifier on the held-out test set. |

---

## 4. Species Class Mapping & Vocabulary Format

### Species Class Ordering (Alphabetical)
Species classifications and confusion matrices use standard **alphabetical zero-based indexing**:

```python
SPECIES_CLASSES = ["Asian Elephant", "Chital", "Lion", "Tiger"]

SPECIES_TO_IDX = {
    "Asian Elephant": 0,
    "Chital": 1,
    "Lion": 2,
    "Tiger": 3,
}

IDX_TO_SPECIES = {0: "Asian Elephant", 1: "Chital", 2: "Lion", 3: "Tiger"}
```

> [!WARNING]
> Note that Member 1's dataset script originally had `Tiger: 0, Lion: 1, Asian Elephant: 2, Chital: 3`. Version 3 training and the saved checkpoints (`wildmatch_v3_species_heads.pth`) use the **alphabetical order above**. Always use `SPECIES_CLASSES` from `models.wildmatch_v3`.

### Vocabulary Format (`wildmatch_v3_vocabulary.json`)
- **Total tokens:** 158
- **Special tokens:**
  - Index `0`: `<PAD>` (Zero-padding token; ignored by text encoder masked mean pooling)
  - Index `1`: `<UNK>` (Out-of-vocabulary fallback)
  - Index `2`: `<BOS>` (Beginning-of-sequence delimiter)
  - Index `3`: `<EOS>` (End-of-sequence delimiter)
- **Sequence Length:** Fixed sequence length of `32` tokens (padded with `0` or truncated with `<EOS>`).

---

## 5. Input Requirements & Preprocessing Specifications

### Image Input Requirements
- **Input Channels:** 3 (RGB). Grayscale and RGBA images are converted to RGB.
- **EXIF Handling:** Orientation auto-corrected using `PIL.ImageOps.exif_transpose`.
- **Canvas Sizing:** $224 \times 224$ pixels.
- **Aspect Ratio Policy:** Aspect ratio is strictly preserved. Images are scaled along their longest side and padded symmetrically with black border `(0, 0, 0)`.
- **Pixel Value Range:** Normalized to $[0.0, 1.0]$ via division by $255.0$.
- **ImageNet Normalization:**
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
- **Tensor Format:** `torch.float32` of shape `[B, 3, 224, 224]`.

### Caption Tokenization Requirements
- **Case:** Lowercased.
- **Punctuation:** Stripped via regex `[^\w\s]`.
- **Whitespace:** Collapsed into single spaces and trimmed.
- **Special Tokens:** Prepended with `<BOS>` (2) and appended with `<EOS>` (3).
- **Padding:** Padded with `<PAD>` (0) up to length 32.
- **Tensor Format:** `torch.long` of shape `[B, 32]`.

---

## 6. Official Experimental Results (Held-Out Test Set, N=30)

The held-out test set contains 30 samples (8 Tigers, 8 Asian Elephants, 7 Lions, 7 Chital):

| Metric | Result | Target / Interpretation |
|---|---:|---|
| **Text-to-Image Species Recall@1** | **100.00%** | Every text query retrieves an image of the correct species at Rank 1 |
| **Text-to-Image Species Recall@5** | **100.00%** | Top-5 retrieved images contain the query species with 100% precision |
| **Image-to-Text Species Recall@1** | **86.67%** | Image query retrieves matching species text at Rank 1 (26/30) |
| **Image-to-Text Species Recall@5** | **90.00%** | Top-5 retrieved captions match the query species (27/30) |
| **Average Species Recall@1** | **93.33%** | Mean bimodal Top-1 species accuracy |
| **Average Species Recall@5** | **95.00%** | Mean bimodal Top-5 species accuracy |
| **Text Classifier Accuracy** | **100.00%** | Auxiliary head correctly classifies all 30 text captions |
| **Image Classifier Accuracy** | **86.67%** | Auxiliary head correctly classifies 26 of 30 test images |
| **Text-to-Image Exact Recall@1** | **16.67%** | Matches exact original image-caption pair at Rank 1 (5/30) |
| **Text-to-Image Exact Recall@5** | **66.67%** | Exact paired image is in Top-5 candidates (20/30) |
| **Image-to-Text Exact Recall@1** | **16.67%** | Matches exact original caption at Rank 1 (5/30) |
| **Image-to-Text Exact Recall@5** | **53.33%** | Exact paired caption is in Top-5 candidates (16/30) |
| **Average Exact Recall@1** | **16.67%** | Mean bimodal Top-1 exact pair match |
| **Average Exact Recall@5** | **60.00%** | Mean bimodal Top-5 exact pair match |

### Critical Retrieval Objective Distinction: Species vs. Exact Recall
- **Species-Level Retrieval (Primary Objective):** Measures whether a retrieved image or caption belongs to the intended wildlife species (e.g., retrieving any Tiger image for a tiger caption). The model achieves **93.33% Recall@1** and **100% Text-to-Image Recall@1**, indicating robust semantic convergence.
- **Exact-Pair Retrieval (Secondary Metric):** Measures whether a query retrieves its single exact paired sample from dataset collection out of 30 candidates. Because multiple images of the same species share almost identical visual attributes (e.g. 8 different tigers resting in grass), multiple tiger captions are semantically valid matches for multiple tiger images. Achieving 66.67% Exact Recall@5 on 30 items demonstrates strong candidate clustering.

---

## 7. Model Loading & Inference Code Examples

### Example 1: Loading the Model and Preprocessing
```python
import torch
from models.inference import (
    load_model,
    load_species_heads,
    preprocess_image,
    tokenize_caption,
    embed_image,
    embed_text,
    compute_similarity,
)

# 1. Load dual encoder (ResNet-18 + BiLSTM)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/wildmatch_v3_dual_encoder.pth", device=device)

# 2. Load auxiliary species classifiers
heads = load_species_heads("models/wildmatch_v3_species_heads.pth", device=device)

# 3. Preprocess image
image_tensor = preprocess_image("path/to/tiger.jpg", device=device)  # [1, 3, 224, 224]

# 4. Tokenize text caption
text_tensor = tokenize_caption("a tiger resting near shallow water", device=device)  # [1, 32]
```

### Example 2: Generating Embeddings & Cross-Modal Similarity
```python
# Generate normalized unit embeddings [1, 256]
img_emb = embed_image(model, image_tensor)
txt_emb = embed_text(model, text_tensor)

# Cosine similarity in [-1.0, 1.0]
sim = compute_similarity(img_emb, txt_emb).item()
print(f"Cross-modal cosine similarity: {sim:.4f}")

# Auxiliary species predictions
_, img_species, img_probs = heads.predict_image(img_emb)
_, txt_species, txt_probs = heads.predict_text(txt_emb)

print(f"Image Species: {img_species[0]} ({img_probs.max().item() * 100:.1f}%)")
print(f"Text Species:  {txt_species[0]} ({txt_probs.max().item() * 100:.1f}%)")
```

### Example 3: Gallery Search (Text-to-Image Retrieval)
```python
# Assuming gallery_embs is [N_gallery, 256] precomputed image embeddings
query_emb = embed_text(model, "herd of elephants walking in woodland") # [1, 256]

# Pairwise dot product gives cosine similarities
sims = torch.matmul(query_emb, gallery_embs.t()).squeeze(0) # [N_gallery]
topk_scores, topk_indices = torch.topk(sims, k=5)

for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
    print(f"Rank {rank}: Gallery Image #{idx.item()} (Score: {score.item():.4f})")
```

---

## 8. Dependencies & Installation Instructions

The Member 3 implementation requires standard scientific and deep learning packages:

```bash
pip install torch torchvision numpy pandas pillow
```

To run all unit tests verifying the architecture and inference components:
```bash
python -m unittest discover tests
```

To run the sample inference demonstration:
```bash
python scripts/sample_inference.py
```

---

## 9. Known Limitations & Recommendations for Member 4

1. **Vocabulary Coverage:** The custom vocabulary contains 158 words curated from the wildlife training captions. Out-of-vocabulary words map to `<UNK>`. For Member 4's Gradio app, users should be prompted with keywords present in the dataset (e.g. *tiger, lion, elephant, chital, deer, water, grass, forest, walking, resting, herd*).
2. **Species Scope:** The model is trained on 4 species (*Asian Elephant, Chital, Lion, Tiger*). Queries outside these 4 species will map to the closest semantic subspace among the 4 known classes.
3. **Exact vs. Species Retrieval in UI:** In the Gradio retrieval interface, Member 4 should highlight the retrieved candidate's species tag as well as its similarity score, so users understand that returning any high-quality tiger image for a tiger caption is a successful retrieval.

