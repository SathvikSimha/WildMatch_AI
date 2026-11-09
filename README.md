# WildMatch AI — Multimodal Wildlife Semantic Matching

**Project:** Fine-Grained Wildlife Image–Text Semantic Matching Using Contrastive Deep Learning  
**Status:** Dataset Pipeline (Member 1) & Dual-Encoder Architecture (Member 2) Fully Implemented & Empirically Verified  
**Dataset Scale:** 200 balanced image–caption pairs across 4 species  
**Model Architecture:** Custom 4-Stage CNN + BiLSTM with Temperature-Calibrated Shared Space  

---

## 1. Project Overview & Division of Responsibilities

WildMatch AI is divided into four modular components developed by four team members, designed to integrate into a cohesive multimodal retrieval system:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    WILDMATCH AI SYSTEM PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. MEMBER 1: DATASET & PREPROCESSING [COMPLETED & VALIDATED]                                    │
│    • iNaturalist API Collection (CC-BY / CC-BY-SA) ──► 200 Curated Images (50/species)          │
│    • BLIP Captioning & Manual Review              ──► 200 Reviewed Captions                     │
│    • Aspect-Preserving Image Standardization      ──► [3, 224, 224] Float32 in [0, 1]           │
│    • Stratified Splitting & Vocab (70/15/15)      ──► 158-word Train-Only vocabulary.json       │
│    • PyTorch Dataset & DataLoader API             ──► WildlifeDataset & create_dataloaders()    │
└────────────────────────────────────────┬────────────────────────────────┬───────────────────────┘
                                         │                                │
                    Image Tensors:       │                                │ Caption Tokens:
                    [B, 3, 224, 224]     │                                │ [B, 32] LongTensor
                                         ▼                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MEMBER 2: DUAL-ENCODER ARCHITECTURE [COMPLETED & VERIFIED]                                   │
│    • ImageEncoder (models/image_encoder.py)       ──► 4-Stage ConvNet + MLP ──► [B, 256] (L2=1) │
│    • TextEncoder (models/text_encoder.py)         ──► BiLSTM + Masked Mean  ──► [B, 256] (L2=1) │
│    • DualEncoder (models/dual_encoder.py)         ──► Calibrated Joint/Unimodal Multimodal Space│
│    • Verification Harness (verify_member2_models) ──► 25/25 Tests Passing (Code 0, 937K Params) │
└────────────────────────────────────────┬────────────────────────────────┬───────────────────────┘
                                         │                                │
                    Symmetric Logits:    │                                │ Unimodal Vectors:
                    get_logits()         │                                │ encode_image / text
                                         ▼                                ▼
┌──────────────────────────────────────────────────┐ ┌────────────────────────────────────────────┐
│ 3. MEMBER 3: CONTRASTIVE TRAINING [HANDOFF READY]│ │ 4. MEMBER 4: RETRIEVAL & APP [HANDOFF READY│
│    • Symmetric InfoNCE Loss (loss.py)            │ │    • Cosine Similarity Matrix (retrieval.py│
│    • AdamW Optimizer & Cosine Annealing          │ │    • Offline Gallery Vector Indexing       │
│    • Contrastive Training Loops (train.py)       │ │    • Top-K Ranking & Metrics (Recall, MRR) │
│    • Loss Curves, Metrics & Model Checkpoints    │ │    • Interactive Gradio Explorer (app.py)  │
└──────────────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

### Responsibility Matrix

| Member | Focus Area | Key Deliverables | Status |
|:---:|---|---|:---:|
| **Member 1** | Dataset Collection & Preprocessing | `dataset/`, `preprocessing/`, `dataset/vocabulary.json`, `scripts/validate_dataloaders.py` | ✅ **Completed & Validated** |
| **Member 2** | Image & Text Encoder Architecture | `models/image_encoder.py`, `models/text_encoder.py`, `models/dual_encoder.py`, `tests/`, `scripts/verify_member2_models.py`, `docs/MODEL_ARCHITECTURE.md` | ✅ **Completed & Verified** |
| **Member 3** | Contrastive Learning & Model Training | `models/wildmatch_v3_dual_encoder.pth`, `models/wildmatch_v3_species_heads.pth`, `models/inference.py`, `MEMBER3_HANDOFF.md`, `docs/MEMBER3_TECHNICAL_REPORT.md` | ✅ **Completed & Delivered (Version 3)** |
| **Member 4** | Similarity Matching, Evaluation & App | `retrieval.py`, `evaluation.py`, `app.py` (Gradio UI), Top-K ranking, Recall@1/5, MRR | 🔄 **Next Step: Use MEMBER3_HANDOFF.md** |

---

## 2. Member 2: Deep Learning Dual-Encoder Architecture

Member 2 designs, implements, and empirically validates the dual-branch neural architecture that maps visual observations and textual descriptions into a shared, calibrated 256-dimensional semantic space.

Full architectural specifications, layer tables, and Mermaid flowcharts are detailed in [**`WildMatch_AI/docs/MODEL_ARCHITECTURE.md`**](docs/MODEL_ARCHITECTURE.md).

> [!NOTE]
> **Important Note on Weight Initialization & Retrieval Scores:**  
> Member 2's deliverable is the **verified deep learning architecture**. Baseline demonstration scripts instantiate `DualEncoder` with default PyTorch random weights (expected random dot products in $\mathbb{R}^{256}$ are near zero: $\sim 0.01 - 0.03$). To achieve high-accuracy semantic retrieval ($+0.65 - +0.85$), **Member 3 must execute the contrastive training pipeline** to produce `checkpoints/best_dual_encoder.pth`.  
> For full mathematical explanation and step-by-step instructions for Member 3 & Member 4, see [**`docs/MEMBER2_HANDOFF_CLARIFICATION.md`**](docs/MEMBER2_HANDOFF_CLARIFICATION.md).

### 2.1 Custom CNN Image Encoder (`models/image_encoder.py`)
- **Backbone**: 4 sequential convolutional stages with 32, 64, 128, and 256 channels.
  - Each stage applies `Conv2d(kernel_size=3, padding=1, bias=False)`, `BatchNorm2d`, `ReLU(inplace=True)`, and `MaxPool2d(2, 2)`.
  - Feature resolution stages: $224 \times 224 \to 112 \times 112 \to 56 \times 56 \to 28 \times 28 \to 14 \times 14$.
- **Spatial Aggregation**: `nn.AdaptiveAvgPool2d((1, 1))` followed by `nn.Flatten()`, yielding a 256-d visual vector invariant to spatial resolution.
- **Projection Head**: Two-layer MLP (`Linear(256, 256)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(256, embed_dim)`).
- **Normalization**: Unit L2 normalization (`F.normalize(p=2, dim=-1)`), ensuring $||v_i||_2 = 1.0 \pm 10^{-5}$.
- **Feature Inspection**: `model.extract_features(images)` exposes intermediate `[B, 256, 14, 14]` spatial activation maps.
- **Parameters**: 520,992 parameters (1.99 MB footprint).

### 2.2 BiLSTM Text Encoder (`models/text_encoder.py`)
- **Word Embedding**: `nn.Embedding(158, 128, padding_idx=0)` with embedding dropout ($p=0.1$).
  - Special token indices: `<PAD>: 0`, `<UNK>: 1`, `<BOS>: 2`, `<EOS>: 3`. Zero gradient guaranteed for padding rows.
- **Recurrent Backbone**: Bidirectional LSTM (`input_size=128`, `hidden_size=128`, 1 layer, `batch_first=True`), producing 256 features per timestep ($[B, 32, 256]$).
- **Sequence Aggregation**: Padding-aware masked mean pooling:
  $$\text{mask} = (\text{tokens} \ne \text{padding\_idx}), \quad \mathbf{h}_{\text{pooled}} = \frac{\sum_t \mathbf{h}_t \odot \text{mask}_t}{\sum_t \text{mask}_t}$$
  Guarantees trailing `<PAD>` tokens do not contaminate sentence semantics.
- **Projection Head**: Two-layer MLP (`Linear(256, 256)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(256, embed_dim)`).
- **Normalization**: Unit L2 normalization, ensuring $||t_i||_2 = 1.0 \pm 10^{-5}$.
- **Parameters**: 416,512 parameters (1.59 MB footprint).

### 2.3 Unified Dual Encoder (`models/dual_encoder.py`)
- **Multimodal Composition**: Unifies `ImageEncoder` and `TextEncoder` with shared projection dimensionality (`embed_dim=256`, configurable to 128 or 512).
- **Temperature Scaling**: Learnable calibration parameter `logit_scale` initialized to $\ln(1 / 0.07) \approx 2.6593$ ($\tau = 0.07$, scale $\approx 14.2857$) and clamped to $\ln(100.0) \approx 4.6052$ for numerical stability.
- **Dual Operating Modes**:
  1. **Joint Training Pass**: `forward(images, captions)` $\to$ `(image_embeds, text_embeds, scale)`
  2. **Calibrated Symmetric Logits**: `get_logits(images, captions)` $\to$ `(logits_per_image, logits_per_text)`
  3. **Unimodal Feature Extraction**: `encode_image(images)` $\to$ `[B, 256]` and `encode_text(captions)` $\to$ `[B, 256]`
  4. **Cosine Similarity Matrix**: `compute_similarity(images, captions)` $\to$ `[B_img, B_txt]` with values strictly in $[-1.0, 1.0]$
- **System Total**: **937,505 parameters** (Memory footprint: **3.58 MB**).

---

## 3. Verification & Testing

Member 2 deliverables include an exhaustive testing hierarchy covering unit tests, standalone invariants, and a master integration runner.

### Running Tests

```bash
# 1. Run Master Verification Suite (Programmatic unittests + Member 1 & 3 & 4 simulations)
python scripts/verify_member2_models.py

# 2. Run all unit test suites (25 tests across image, text, and dual encoders)
python -m unittest discover tests -v

# 3. Run individual model verification and profiling scripts
python scripts/verify_image_encoder.py
python scripts/verify_text_encoder.py
python scripts/verify_dual_encoder.py

# 4. Verify Member 1 DataLoader integrity (15 checks)
python scripts/validate_dataloaders.py
```

### Master Verification Results (`verify_member2_models.py`)

```text
================================================================================
           WILDMATCH AI - MEMBER 2 MASTER MODEL VERIFICATION SUITE              
================================================================================
[PASS] STAGE 1: Programmatic Unit Test Suites (25 tests run, 0 failures, 0 errors)
[PASS] STAGE 2: Member 1 Data Pipeline Integration (Vocabulary + DataLoader tensors)
[PASS] STAGE 3: Simulated Member 3 Contrastive Training Step (InfoNCE + AdamW updates)
[PASS] STAGE 4: Simulated Member 4 Retrieval Step (Cosine Ranking + Recall/MRR)
[PASS] STAGE 5: Structural & Performance Audit (937,505 params, 3.58 MB, ~19 pairs/sec)
================================================================================
[PASS] ALL MEMBER 2 MODEL VERIFICATIONS SUCCESSFUL (Exit Code 0)
================================================================================
```

---

## 4. Downstream Integration Guides

### 4.1 For Member 3 (Contrastive Learning & Training)

Member 3 can instantiate `DualEncoder` and directly compute symmetric cross-entropy contrastive loss:

```python
import torch
import torch.nn.functional as F
from WildMatch_AI.models import DualEncoder
from WildMatch_AI.dataset.dataloader import create_dataloaders

# 1. Load DataLoaders and instantiate DualEncoder
train_loader, val_loader, test_loader = create_dataloaders(batch_size=16)
model = DualEncoder(embed_dim=256, init_temperature=0.07)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# 2. Training Step
model.train()
for batch in train_loader:
    images, captions = batch['image'], batch['caption']
    
    # get_logits returns temperature-scaled logits [B, B]
    logits_per_image, logits_per_text = model.get_logits(images, captions)
    
    # Ground-truth diagonal targets
    targets = torch.arange(images.size(0), device=images.device)
    loss = (F.cross_entropy(logits_per_image, targets) + 
            F.cross_entropy(logits_per_text, targets)) / 2.0
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 4.2 For Member 4 (Retrieval, Evaluation & Gradio UI)

Member 4 can extract unimodal embeddings, compute cosine similarities, rank candidates, and build an interactive interface:

```python
import torch
from WildMatch_AI.models import DualEncoder

model = DualEncoder(embed_dim=256)
model.eval()

# 1. Unimodal Extraction
with torch.no_grad():
    gallery_image_embeddings = model.encode_image(all_gallery_images)  # [N_img, 256] (L2 norm = 1.0)
    query_text_embedding = model.encode_text(query_caption_tokens)     # [1, 256] (L2 norm = 1.0)

# 2. Cosine Similarity & Top-K Ranking
# Dot product of L2-normalized vectors directly equals cosine similarity
similarity_scores = torch.matmul(query_text_embedding, gallery_image_embeddings.t())  # [1, N_img]
top5_scores, top5_indices = torch.topk(similarity_scores, k=5, dim=-1)

# top5_indices contains rank-ordered gallery indices for display in Gradio UI
```

---

## 5. Member 3: Contrastive Learning, Model Training & Version 3 Deliverables

Member 3 owns the contrastive learning formulation, loss design, auxiliary supervision, hyperparameter scheduling, training execution, model checkpoint selection, and experimental evaluation.

Full technical details, training curves, and mathematical formulations are documented in [**`docs/MEMBER3_TECHNICAL_REPORT.md`**](docs/MEMBER3_TECHNICAL_REPORT.md). For downstream integration instructions for Member 4, see [**`MEMBER3_HANDOFF.md`**](MEMBER3_HANDOFF.md).

### 5.1 Version 3 Model Architecture
- **Visual Backbone:** ImageNet-pretrained ResNet-18 (512-d features exiting Layer 4). Fine-tuned with conservative learning rate ($\eta_{\text{backbone}} = 10^{-5}$). Zero remote weight downloads required for inference.
- **Text Encoder:** Dense word embedding ($158 \to 128$) + single-layer BiLSTM ($128$ hidden per direction $\to 256$-d) with padding-aware masked mean pooling across non-PAD tokens.
- **Projection Heads:** Visual ($512 \to 512 \to 256$) and Textual ($256 \to 256 \to 256$) 2-layer MLPs mapping to $D=256$.
- **Normalization:** Unit L2 hypersphere normalization ($\|\mathbf{v}\|_2 = 1.0, \|\mathbf{t}\|_2 = 1.0 \in \mathbb{S}^{255}$).
- **Species-Aware Multi-Positive Contrastive Loss:** Multi-positive InfoNCE grouping all images and captions sharing the same biological species label as positive pairs, eliminating the false negative problem of standard InfoNCE.
- **Auxiliary Species Supervision:** Cross-entropy species classification heads on both image and text embeddings (`Linear(256 -> 4)`).

### 5.2 Version 3 Experimental Results (Held-Out Test Set, $N=30$)

| Metric | Result | Analysis |
|---|---:|---|
| **Text-to-Image Species Recall@1** | **100.00%** | All 30 text queries retrieve the correct species at Rank 1 |
| **Text-to-Image Species Recall@5** | **100.00%** | All 30 text queries retrieve the correct species in Top-5 |
| **Image-to-Text Species Recall@1** | **86.67%** | 26 of 30 image queries retrieve matching species text at Rank 1 |
| **Image-to-Text Species Recall@5** | **90.00%** | 27 of 30 image queries retrieve matching species text in Top-5 |
| **Text Classifier Accuracy** | **100.00%** | Auxiliary head correctly classifies all 30 text captions |
| **Image Classifier Accuracy** | **86.67%** | Auxiliary head correctly classifies test images (26/30) |
| **Average Species Recall@1** | **93.33%** | Mean bimodal species retrieval accuracy |
| **Average Species Recall@5** | **95.00%** | Mean bimodal Top-5 species accuracy |
| **Text-to-Image Exact Recall@1** | **16.67%** | Exact original pair retrieved at Rank 1 (5/30) |
| **Text-to-Image Exact Recall@5** | **66.67%** | Exact original pair retrieved in Top-5 (20/30) |
| **Image-to-Text Exact Recall@1** | **16.67%** | Exact original pair retrieved at Rank 1 (5/30) |
| **Image-to-Text Exact Recall@5** | **53.33%** | Exact original pair retrieved in Top-5 (16/30) |
| **Average Exact Recall@1** | **16.67%** | Mean bimodal Top-1 exact retrieval |
| **Average Exact Recall@5** | **60.00%** | Mean bimodal Top-5 exact retrieval |

### 5.3 Model Artifacts & Deliverables
All deliverables are archived in `models/` and root:
- `models/wildmatch_v3_dual_encoder.pth`: Trained state dict of `WildMatchV3DualEncoder` (48.0 MB).
- `models/wildmatch_v3_species_heads.pth`: Trained auxiliary classification heads (11.1 KB).
- `models/wildmatch_v3_vocabulary.json`: 158-word vocabulary mapping with special tokens.
- `models/wildmatch_v3_metrics.json`: Evaluated test metrics.
- `models/wildmatch_v3_history.json` & `history.csv`: 34-epoch training loss and validation progression.
- Training loss and validation retrieval graphs: `models/wildmatch_v3_loss.png`, `models/wildmatch_v3_validation.png`.
- Test set confusion matrices: `models/wildmatch_v3_image_confusion.png`, `models/wildmatch_v3_text_confusion.png`.
- Reusable inference API: [`models/inference.py`](models/inference.py).
- Sample inference demonstration script: [`scripts/sample_inference.py`](scripts/sample_inference.py).

---

## 6. Member 1: Dataset Setup & Extraction

Image binary files are hosted separately on Google Drive to keep the git repository lightweight (< 1 MB). All captions, vocabulary, split manifests, and scripts are tracked directly in git.

### Google Drive Download Links

[**Download WildMatch AI Dataset Archives on Google Drive**](https://drive.google.com/drive/folders/13XJRsQHok8HqZi3kYecnlEcWoTWI4gVY?usp=sharing)

- **`wildmatch_curated_dataset.zip`** (~70 MB): 200 curated, verified wildlife images (50/species). **Required for training and DataLoader image loading.**
- **`wildmatch_raw_dataset.zip`** (~100 MB): 280 raw collected images from iNaturalist before curation (optional).

### Extraction Instructions

1. Download `wildmatch_curated_dataset.zip` and extract into `dataset/`:
   - `dataset/curated/tiger/` (50 images)
   - `dataset/curated/lion/` (50 images)
   - `dataset/curated/elephant/` (50 images)
   - `dataset/curated/chital/` (50 images)
2. Run DataLoader validation: `python scripts/validate_dataloaders.py`.

### Dataset Distribution

| Species | Train (70%) | Validation (15%) | Test (15%) | Total Curated | Licence Share |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tiger** (*Panthera tigris*) | 35 | 7 | 8 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Lion** (*Panthera leo*) | 35 | 8 | 7 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Asian Elephant** (*Elephas maximus*) | 35 | 7 | 8 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Chital** (*Axis axis*) | 35 | 8 | 7 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Total** | **140** | **30** | **30** | **200** | **100% Balanced** |

- **Duplicate / Leakage Audit**: 0 cross-split duplicates, 0 byte-level duplicates. 1 documented near-duplicate Tiger pair isolated within train split.
- **Vocabulary**: 158 words fitted strictly on training split (`dataset/vocabulary.json`). Held-out OOV rate: 3.8%.

---

## 6. Repository Structure

```text
WildMatch_AI/
├── .gitignore
├── README.md                             # Comprehensive project documentation
├── requirements.txt                      # Project dependencies (Python 3.9+)
├── docs/
│   ├── MODEL_ARCHITECTURE.md             # Member 2 architecture specification & handoff
│   └── MEMBER2_HANDOFF_CLARIFICATION.md  # Architectural status & downstream training handoff guide
├── models/                               # Member 2: Deep Learning Neural Architectures
│   ├── __init__.py                       # Package exports (ImageEncoder, TextEncoder, DualEncoder)
│   ├── image_encoder.py                  # Custom 4-stage CNN visual feature extractor
│   ├── text_encoder.py                   # BiLSTM caption encoder with masked pooling
│   └── dual_encoder.py                   # Unified multimodal dual encoder & calibration
├── tests/                                # Comprehensive Unit Test Suites
│   ├── __init__.py
│   ├── test_image_encoder.py             # 7 unit tests for ImageEncoder
│   ├── test_text_encoder.py              # 9 unit tests for TextEncoder
│   └── test_dual_encoder.py              # 9 unit tests for DualEncoder
├── dataset/                              # Member 1: Multimodal Data Ingestion Pipeline
│   ├── dataloader.py                     # PyTorch DataLoader factory
│   ├── wildlife_dataset.py               # PyTorch WildlifeDataset implementation
│   ├── vocabulary.json                   # 158-token training vocabulary
│   ├── metadata.csv                      # Raw collection metadata (280 records)
│   ├── collection_report.csv             # API harvest report
│   ├── captions/
│   │   ├── pilot_captions.csv            # 200 BLIP and reviewed captions
│   │   └── processed_captions.csv        # Cleaned captions & tokens
│   ├── curation/
│   │   ├── curation_status.csv           # Curation decisions
│   │   └── README.md                     # Curation documentation
│   ├── splits/
│   │   ├── train.csv                     # 140 training pairs
│   │   ├── validation.csv                # 30 validation pairs
│   │   ├── test.csv                      # 30 test pairs
│   │   └── split_manifest.csv            # Assignment manifest
│   └── curated/                          # Extracted from wildmatch_curated_dataset.zip
│       ├── tiger/ (50 images)
│       ├── lion/ (50 images)
│       ├── elephant/ (50 images)
│       └── chital/ (50 images)
├── preprocessing/                        # Member 1: Data Preparation Modules
│   ├── collect_inaturalist.py            # iNaturalist API collector
│   ├── curate_dataset.py                 # Visual contact sheet curation
│   ├── finalize_expansion_curated.py     # Dataset sync utility
│   ├── generate_captions.py              # BLIP caption generator
│   ├── image_preprocessing.py            # Aspect-preserving standardizer [3, 224, 224]
│   └── text_preprocessing.py             # Caption cleaner & tokenizer (len=32)
└── scripts/                              # Verification, Validation & Build Utilities
    ├── verify_member2_models.py          # Master 5-stage Member 2 verification suite
    ├── verify_image_encoder.py           # ImageEncoder profiling & invariant checks
    ├── verify_text_encoder.py            # TextEncoder profiling & invariant checks
    ├── verify_dual_encoder.py            # DualEncoder profiling & invariant checks
    ├── validate_dataloaders.py           # Member 1 15-point DataLoader validation
    ├── split_dataset.py                  # Stratified train/val/test splitter
    └── build_vocabulary.py               # Training-only vocabulary builder
```

---

## 7. Quick Command Reference

```powershell
# Install project dependencies
pip install -r requirements.txt

# Run all Member 2 unit tests
python -m unittest discover tests -v

# Run Member 2 Master Verification Suite
python scripts/verify_member2_models.py

# Run Member 1 DataLoader Validation Suite
python scripts/validate_dataloaders.py
```

---

*Last Updated: 2026-11-08*  
*Contributors: WildMatch AI Team (Members 1, 2, 3, 4)*
