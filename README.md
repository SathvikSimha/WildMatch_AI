# WildMatch AI — Multimodal Wildlife Semantic Matching & Retrieval

**Project:** Fine-Grained Wildlife Image–Text Semantic Matching and Retrieval Using Contrastive Deep Learning  
**Status:** **Fully Implemented, Trained, Evaluated, and Verified (Members 1, 2, 3, and 4 Complete)**  
**Dataset Scale:** 200 curated image–caption pairs across 4 focal wildlife species  
**Model Architecture:** ResNet-18 visual backbone + BiLSTM textual encoder with 256-dimensional unit $L_2$-normalized shared semantic space  

---

## 1. Project Overview & Division of Responsibilities

WildMatch AI is an end-to-end multimodal deep learning system designed for biodiversity and wildlife observation matching. The system maps raw visual imagery and natural language descriptions into a calibrated, shared 256-dimensional hypersphere $\mathbb{S}^{255}$, enabling bidirectional cross-modal retrieval via cosine similarity.

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
│    • Architectural Invariants & Shape Verification──► 25/25 Tests Passing (937K Params)        │
└────────────────────────────────────────┬────────────────────────────────┬───────────────────────┘
                                         │                                │
                    Feature Tensors &    │                                │ Symmetric Logits &
                    Embedding Spaces     │                                │ Projection Heads
                                         ▼                                ▼
┌──────────────────────────────────────────────────┐ ┌────────────────────────────────────────────┐
│ 3. MEMBER 3: CONTRASTIVE TRAINING [COMPLETED]      │ │ 4. MEMBER 4: RETRIEVAL & APP [COMPLETED]   │
│    • Version 3 ResNet-18 + BiLSTM (wildmatch_v3) │ │    • Cosine Similarity Matrix (retrieval.py│
│    • Symmetric InfoNCE & Multi-Positive Loss     │ │    • Gallery Indexing & Cross-Modal Top-K  │
│    • Auxiliary Species Supervision (Linear 256->4│ │    • Evaluation Suite (Recall@K, MRR)      │
│    • Training Pipeline & Scheduling (train.py)   │ │    • Interactive Gradio Explorer (app.py)  │
│    • Checkpoints: wildmatch_v3_dual_encoder.pth  │ │    • Evaluation Report & Dashboard Cards   │
└──────────────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

### Responsibility Matrix

| Member | Focus Area | Key Deliverables | Status |
|:---:|---|---|:---:|
| **Member 1** | Dataset Collection & Preprocessing | `dataset/`, `preprocessing/`, `dataset/vocabulary.json`, `scripts/validate_dataloaders.py` | ✅ **Completed & Validated** |
| **Member 2** | Image & Text Encoder Architecture | `models/image_encoder.py`, `models/text_encoder.py`, `models/dual_encoder.py`, `tests/test_dual_encoder.py`, `docs/MODEL_ARCHITECTURE.md` | ✅ **Completed & Verified** |
| **Member 3** | Contrastive Learning & Model Training | `training/loss.py`, `training/train.py`, `models/wildmatch_v3.py`, `models/wildmatch_v3_dual_encoder.pth`, `docs/MEMBER3_TECHNICAL_REPORT.md` | ✅ **Completed & Delivered** |
| **Member 4** | Similarity Matching, Evaluation & App | `retrieval.py`, `evaluation.py`, `app.py` (Gradio UI), `docs/member4_evaluation_results.json`, `tests/test_member4_retrieval_eval.py` | ✅ **Completed & Verified** |

---

## 2. Member 1: Dataset Setup & Preprocessing

The dataset comprises 200 curated wildlife observations across four species: **Tiger** (*Panthera tigris*), **Lion** (*Panthera leo*), **Asian Elephant** (*Elephas maximus*), and **Chital** (*Axis axis*).

### Dataset Distribution

| Species | Train (70%) | Validation (15%) | Test (15%) | Total Curated | License Share |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tiger** | 35 | 7 | 8 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Lion** | 35 | 8 | 7 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Asian Elephant** | 35 | 7 | 8 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Chital** | 35 | 8 | 7 | 50 | CC-BY (91%) / CC-BY-SA (9%) |
| **Total** | **140** | **30** | **30** | **200** | **100% Balanced** |

### Preprocessing Pipelines
- **Visual Standardization (`preprocessing/image_preprocessing.py`):**
  1. Auto-rotation via `PIL.ImageOps.exif_transpose` for sensor orientation.
  2. RGB channel enforcement (strips alpha, expands grayscale).
  3. Aspect-preserving resize with symmetric black letterboxing to $224 \times 224$.
  4. Conversion to Float32 $[0.0, 1.0]$ with ImageNet mean/std normalization.
- **Text Standardization (`preprocessing/text_preprocessing.py`):**
  1. Lowercased alphanumeric cleaning removing punctuation.
  2. Vocabulary mapping with special tokens (`<PAD>: 0`, `<UNK>: 1`, `<BOS>: 2`, `<EOS>: 3`).
  3. Sequence standardization to fixed length $L=32$ tokens.
  4. Built strictly on the training split ($N=140$) to prevent data leakage.

### Dataset Archives on Google Drive
Image binary files are hosted separately on Google Drive to keep the repository lightweight:
- [**Download WildMatch AI Dataset on Google Drive**](https://drive.google.com/drive/folders/13XJRsQHok8HqZi3kYecnlEcWoTWI4gVY?usp=sharing)
- Extract `wildmatch_curated_dataset.zip` into `dataset/curated/` (`tiger/`, `lion/`, `elephant/`, `chital/`).

---

## 3. Member 2: Dual-Encoder Neural Architecture

Member 2 designed the custom dual-encoder architecture (`models/dual_encoder.py`) mapping images and text into a shared 256-dimensional semantic space:

- **Custom CNN Image Encoder (`models/image_encoder.py`):** 4-stage convolutional backbone (32 $\to$ 64 $\to$ 128 $\to$ 256 channels), adaptive average pooling, and 2-layer MLP projection head (520K parameters).
- **BiLSTM Text Encoder (`models/text_encoder.py`):** Word embedding ($158 \to 128$), single-layer bidirectional LSTM ($128$ hidden per direction), padding-aware masked mean pooling, and 2-layer MLP projection head (416K parameters).
- **Dual Encoder (`models/dual_encoder.py`):** Composes image and text branches with learnable temperature scaling parameter `logit_scale` initialized to $\ln(1 / 0.07) \approx 2.6593$.

Full architectural specifications are detailed in [**`docs/MODEL_ARCHITECTURE.md`**](docs/MODEL_ARCHITECTURE.md).

---

## 4. Member 3: Contrastive Learning & Model Training

Member 3 developed the training objectives, contrastive losses, auxiliary species supervision, and model training pipeline.

### 4.1 Contrastive Loss Modules (`training/loss.py`)
1. **Symmetric Contrastive Loss (`SymmetricContrastiveLoss`):**
   Standard bidirectional InfoNCE / NT-Xent:
   $$\mathcal{L}_{\text{sym}} = \frac{1}{2} \left( \mathcal{L}_{v \to t} + \mathcal{L}_{t \to v} \right)$$
   Where each direction is categorical cross-entropy against diagonal ground-truth identity targets scaled by temperature $\tau$.
2. **Species-Aware Multi-Positive Contrastive Loss (`SpeciesAwareMultiPositiveContrastiveLoss`):**
   Mitigates the false negative penalty of standard InfoNCE by treating all in-batch pairs sharing the same species label as positive pairs:
   $$\mathcal{L}_{\text{multi-pos}}^{(v \to t)} = -\frac{1}{B} \sum_{i=1}^B \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{v}_i \cdot \mathbf{t}_p / \tau)}{\sum_{k=1}^B \exp(\mathbf{v}_i \cdot \mathbf{t}_k / \tau)}$$
3. **Auxiliary Species Supervision (`AuxiliaryClassificationLoss`):**
   Linear classification heads on 256-d embeddings (`Linear(256 -> 4)`) trained with cross-entropy loss:
   $$\mathcal{L}_{\text{aux}} = \frac{1}{2} \left( \text{CE}(\hat{\mathbf{y}}_v, \mathbf{y}) + \text{CE}(\hat{\mathbf{y}}_t, \mathbf{y}) \right)$$
4. **Unified Objective (`WildMatchLoss`):**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{contrastive}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}, \quad \lambda_{\text{aux}} = 0.5$$

### 4.2 End-to-End Training Pipeline (`training/train.py`)
- **Optimizer:** AdamW with differential learning rates ($\eta_{\text{backbone}} = 10^{-5}$, $\eta_{\text{heads}} = 2 \times 10^{-4}$, weight decay $10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` monitoring validation species Recall@1.
- **Checkpoints:** Saves best checkpoint (`wildmatch_v3_best.pth`), delivery models (`wildmatch_v3_dual_encoder.pth`, `wildmatch_v3_species_heads.pth`), and full training history CSV/JSON.

```powershell
# Run model training from CLI
python training/train.py --epochs 35 --batch-size 16 --lr-backbone 1e-5 --lr-heads 2e-4
```

### 4.3 Training Results & Benchmark Metrics (Held-Out Test Set, $N=30$)

| Evaluation Metric | Exact Match Paradigm | Species-Level Semantic Paradigm |
|---|:---:|:---:|
| **Text-to-Image Recall@1** | 16.67% (5/30) | **100.00%** (30/30) |
| **Text-to-Image Recall@5** | 66.67% (20/30) | **100.00%** (30/30) |
| **Text-to-Image Recall@10** | 96.67% (29/30) | **100.00%** (30/30) |
| **Text-to-Image MRR** | 0.3650 | **1.0000** |
| **Image-to-Text Recall@1** | 16.67% (5/30) | **83.33%** (25/30) |
| **Image-to-Text Recall@5** | 53.33% (16/30) | **90.00%** (27/30) |
| **Image-to-Text Recall@10** | 90.00% (27/30) | **100.00%** (30/30) |
| **Image-to-Text MRR** | 0.3465 | **0.8792** |
| **Bimodal Average Recall@1** | 16.67% | **91.67%** |
| **Bimodal Average Recall@5** | 60.00% | **95.00%** |
| **Bimodal Average Recall@10** | 93.33% | **100.00%** |
| **Bimodal Average MRR** | 0.3557 | **0.9396** |
| **Image Classifier Accuracy** | — | **86.67%** (26/30) |
| **Text Classifier Accuracy** | — | **100.00%** (30/30) |
| **Species Separation Margin** | — | **+0.6158** (Same: +0.429, Diff: -0.187) |

---

## 5. Member 4: Retrieval Engine, Evaluation Suite & Gradio App

Member 4 delivered the downstream similarity matching, evaluation benchmark suite, and interactive web application.

### 5.1 Retrieval Engine (`retrieval.py`)
- **`calculate_cosine_similarity`:** Numerically clamped dot product on unit $L_2$-normalized vectors.
- **`WildMatchRetriever`:** Offline vector gallery indexer supporting split filtering (`train`, `validation`, `test`, `all`) and biological species filtering.
- **Bidirectional Endpoints:** `retrieve_images_by_text()` and `retrieve_texts_by_image()`.
- **Cross-Modal Pair Scorer:** `score_image_text_pair()` providing cosine similarity and dual-branch species prediction probabilities.

```powershell
# Run retrieval query from CLI
python retrieval.py --split test --query "a tiger resting near shallow water" --top-k 5
```

### 5.2 Evaluation Suite (`evaluation.py`)
Computes comprehensive quantitative benchmarks across splits:
- Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR), Mean First Rank.
- Per-species breakdown, similarity margins, and auxiliary classification accuracies.
- Exports structured evaluation JSON reports.

```powershell
# Run full evaluation benchmark
python evaluation.py --split test --save-json docs/member4_evaluation_results.json
```

### 5.3 Interactive Gradio Application (`app.py`)
A production-ready web application providing 4 functional interfaces:

1. **Text-to-Image Retrieval Tab:** Natural language query input, Top-K slider (1-10), species filter dropdown, gallery display with similarity scores, and metadata inspection table.
2. **Image-to-Text Retrieval Tab:** Wildlife image uploader with sample test buttons, predicted image species probabilities, and Top-K ranked descriptive captions.
3. **Pair Similarity Scorer Tab:** Multi-modal pair evaluation showing visual cosine similarity meter, match assessment banner, and dual species predictions.
4. **Evaluation & Analytics Dashboard:** Split performance KPIs, aggregate retrieval metrics, per-species breakdown table, training loss curves, and test confusion matrices.

```powershell
# Launch Gradio web application locally
python app.py

# Launch with temporary public share link
python app.py --share
```

---

## 6. Repository Structure

```text
WildMatch_AI/
├── .gitignore                             # Git exclusion specifications
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Project dependencies
├── app.py                                 # Member 4: Interactive Gradio web application
├── retrieval.py                           # Member 4: Multimodal retrieval engine
├── evaluation.py                          # Member 4: Evaluation & metrics suite
├── training/                              # Member 3: Training & Contrastive Loss Package
│   ├── __init__.py                        # Package exports
│   ├── loss.py                            # Symmetric & Multi-Positive InfoNCE loss
│   └── train.py                           # End-to-end model training pipeline
├── models/                                # Deep Learning Architectures & Checkpoints
│   ├── __init__.py                        # Model exports & version loaders
│   ├── image_encoder.py                   # Member 2: Custom CNN visual feature extractor
│   ├── text_encoder.py                    # Member 2: BiLSTM caption encoder
│   ├── dual_encoder.py                    # Member 2: Calibrated dual encoder
│   ├── wildmatch_v3.py                    # Member 3: Version 3 ResNet-18 + BiLSTM
│   ├── inference.py                       # Member 3: High-level inference utilities
│   ├── wildmatch_v3_dual_encoder.pth      # Member 3: Trained dual encoder weights (45.8 MB)
│   ├── wildmatch_v3_species_heads.pth     # Member 3: Trained auxiliary classification heads
│   ├── wildmatch_v3_vocabulary.json       # 158-word vocabulary mapping
│   ├── wildmatch_v3_metrics.json          # Test split benchmark metrics
│   ├── wildmatch_v3_history.json          # 34-epoch training loss history
│   ├── wildmatch_v3_loss.png              # Training and validation loss curves
│   └── wildmatch_v3_validation.png        # Validation Recall@1 curve
├── dataset/                               # Member 1: Multimodal Data Ingestion Pipeline
│   ├── dataloader.py                      # PyTorch DataLoader factory
│   ├── wildlife_dataset.py                # PyTorch WildlifeDataset implementation
│   ├── vocabulary.json                    # 158-token training vocabulary
│   ├── metadata.csv                       # Curated dataset metadata (200 records)
│   ├── splits/                            # Stratified dataset split manifests
│   │   ├── train.csv                      # 140 training pairs
│   │   ├── validation.csv                 # 30 validation pairs
│   │   └── test.csv                       # 30 test pairs
│   └── curated/                           # Curated wildlife images (50/species)
├── preprocessing/                         # Member 1: Data Preparation Modules
│   ├── image_preprocessing.py             # Aspect-preserving standardizer [3, 224, 224]
│   └── text_preprocessing.py              # Caption cleaner & tokenizer (len=32)
├── tests/                                 # Unit & Integration Test Suites
│   ├── test_dual_encoder.py               # 9 tests for DualEncoder
│   ├── test_image_encoder.py              # 7 tests for ImageEncoder
│   ├── test_text_encoder.py               # 9 tests for TextEncoder
│   ├── test_v3_inference.py               # 5 tests for Version 3 inference
│   ├── test_training.py                   # 8 tests for contrastive losses & optimizer
│   └── test_member4_retrieval_eval.py     # Member 4 retrieval, evaluation, and UI tests
├── docs/                                  # Technical Reports & Architecture Documentation
│   ├── MODEL_ARCHITECTURE.md              # Member 2 architecture specifications
│   ├── MEMBER3_TECHNICAL_REPORT.md        # Member 3 training & loss formulation report
│   └── member4_evaluation_results.json    # Full test split evaluation benchmark
└── scripts/                               # Reproducibility & Verification Scripts
    ├── sample_inference.py                # Quick terminal inference demonstration
    ├── test_v3_reproducibility.py         # Version 3 metric reproducibility script
    ├── validate_dataloaders.py            # Member 1 DataLoader validation
    └── verify_member2_models.py           # Member 2 architecture verification
```

---

## 7. Execution Guide: Windows 11 & Google Colab

### 7.1 Local Windows 11 Environment

```powershell
# 1. Clone repository and navigate to workspace
cd "d:\College\Deep Learning\WildMatch_AI"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all unit tests (All test suites pass)
python -m unittest discover tests -v

# 4. Run sample terminal inference demo (No images required)
python scripts/sample_inference.py

# 5. Execute model training pipeline (Optional)
python training/train.py --epochs 35 --batch-size 16

# 6. Launch the Gradio web application
python app.py
```

### 7.2 Google Colab (Free GPU / T4)

```python
# 1. Clone the repository
!git clone https://github.com/<your-username>/WildMatch_AI.git
%cd WildMatch_AI

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run test suite
!python -m unittest discover tests -v

# 4. Launch Gradio with public shareable URL
!python app.py --share
```

---

## 8. Limitations & Future Enhancements

### Limitations
1. **Curated Vocabulary Size (158 tokens):** While effective for specific wildlife species and environmental contexts, out-of-vocabulary words in user queries map to `<UNK>`.
2. **Species Scope (4 Classes):** Optimized for four flagship species (Tiger, Lion, Asian Elephant, Chital). Unseen species require additional curated data.
3. **Exact-Pair vs. Semantic Multi-Positive Retrieval:** Due to multiple images sharing similar captions per species, semantic species retrieval is exceptionally high (100% Recall@1), whereas exact identity matching operates within a multi-candidate set.

### Future Enhancements
- Expand species taxonomy to 20+ terrestrial mammals using hierarchical species supervision.
- Incorporate subword BPE tokenization (WordPiece or Byte-Pair Encoding) to eliminate `<UNK>` occurrences.
- Integrate FAISS vector index for real-time sub-millisecond retrieval on large-scale galleries (>100,000 images).

---

*Academic Final Submission: WildMatch AI Project Team (Members 1, 2, 3, 4)*  
*Department of Computer Science & Engineering — Deep Learning Coursework*
