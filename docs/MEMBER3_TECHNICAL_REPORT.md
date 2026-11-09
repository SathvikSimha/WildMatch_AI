# WildMatch AI — Member 3 Technical Report: Contrastive Learning, Multi-Positive Objectives & Model Training

**Project Title:** WildMatch AI: Wildlife Image-Text Semantic Matching and Retrieval  
**Module:** Member 3 — Contrastive Learning, Multimodal Training & Checkpoint Finalization  
**Author:** Senior Deep Learning Engineer (Member 3)  
**Date:** November 2026  
**Status:** Version 3 Completed, Validated, and Archived for Production Integration  

---

## 1. Problem Statement

Wildlife monitoring and ecological research require indexing, searching, and categorizing large multimodal datasets consisting of field camera-trap imagery, conservation photography, and accompanying observational descriptions. Traditional keyword search systems fail when queries contain descriptive nuances (e.g. *"tiger resting near shallow river"*) or when unstructured text must be matched directly against raw visual scenes.

The goal of **WildMatch AI** is to build a bidirectional multimodal semantic retrieval system capable of:
1. **Text-to-Image Retrieval:** Finding matching wildlife images given descriptive natural language captions.
2. **Image-to-Text Retrieval:** Finding the most descriptive, accurate captions and species annotations given an arbitrary wildlife image.
3. **Cross-Modal Semantic Alignment:** Embedding visual and textual modalities into a shared $D=256$ dimensional hypersphere $\mathbb{S}^{255}$ such that true semantic pairs exhibit high cosine similarity while non-matching pairs are separated.
4. **Species Grounding:** Ensuring that cross-modal retrieval consistently respects biological species boundaries across four focal wildlife species: **Asian Elephant**, **Chital**, **Lion**, and **Tiger**.

Member 3 owns the contrastive learning formulation, loss design, auxiliary supervision, hyperparameter scheduling, training execution, model checkpoint selection, and experimental evaluation.

---

## 2. Dataset Description & Split Sizes

The dataset comprises 200 carefully curated wildlife image-caption pairs across four primary target species collected via the iNaturalist platform and annotated with descriptive BLIP-generated and manually curated captions.

The dataset is partitioned into three disjoint subsets with stratified species representation:

| Split | Total Samples | Asian Elephant | Chital | Lion | Tiger | Percentage | Purpose |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Train** | 140 | 35 | 35 | 35 | 35 | 70.0% | Model optimization & vocabulary construction |
| **Validation** | 30 | 7 | 8 | 8 | 7 | 15.0% | Model selection & hyperparameter tracking |
| **Test** (Held-Out) | 30 | 8 | 7 | 7 | 8 | 15.0% | Final unbiased reporting & benchmark evaluation |
| **Total** | **200** | **50** | **50** | **50** | **50** | **100.0%** | Complete curated benchmark |

### Data Separation Integrity
- The vocabulary of 158 words is built strictly on the **training set** to prevent data leakage.
- Validation and test splits were held completely out of gradient updates and vocabulary construction.
- Final experimental metrics are computed exclusively on the held-out test split ($N=30$).

---

## 3. Dataset Preprocessing

### 3.1 Visual Preprocessing Pipeline
Raw wildlife images exhibit varied aspect ratios, resolutions, and orientations. The visual preprocessing pipeline standardizes all images without geometric distortion:
1. **EXIF Correction:** Auto-rotation using `PIL.ImageOps.exif_transpose` ensures that portrait images captured by vertical sensors are oriented properly.
2. **RGB Enforcement:** Channels are standardized to 3-channel RGB, converting RGBA (alpha stripped) or grayscale images.
3. **Aspect-Preserving Resize:** The image is scaled along its longest dimension to fit inside a $224 \times 224$ bounding box:
   $$s = \min\left(\frac{224}{W}, \frac{224}{H}\right), \quad W' = \lfloor W \cdot s \rceil, \quad H' = \lfloor H \cdot s \rceil$$
4. **Symmetric Zero-Padding (Letterboxing):** The remaining margins are symmetrically padded with black borders `(0, 0, 0)`, placing the subject at the exact optical center:
   $$\text{pad}_{\text{left}} = \frac{224 - W'}{2}, \quad \text{pad}_{\text{top}} = \frac{224 - H'}{2}$$
5. **Tensor Conversion & ImageNet Normalization:** Pixel integers in $[0, 255]$ are scaled to float32 $[0.0, 1.0]$ and normalized using standard ImageNet parameters:
   $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}, \quad \boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$

### 3.2 Caption Preprocessing & Tokenization
Textual descriptions are normalized and tokenized to handle natural variations:
1. **Case & Punctuation Cleaning:** Text is lowercased and stripped of special characters via regex `[^\w\s]`, preserving alphanumeric tokens and standard spacing.
2. **Vocabulary Numericalization:** Words are mapped to integers using the 158-word vocabulary. Out-of-vocabulary tokens map to `<UNK>` (index 1).
3. **Special Delimiters:** Every sequence begins with `<BOS>` (index 2) and terminates with `<EOS>` (index 3).
4. **Padding & Truncation:** Sequences are standardized to $L=32$ tokens using `<PAD>` (index 0).

---

## 4. ResNet-18 Image Encoder

Version 3 replaces the shallow scratch CNN with an **ImageNet-pretrained ResNet-18** convolutional backbone, providing robust hierarchical visual representations (edges, textures, object parts, animal silhouettes).

### Layer Configuration
- **Input:** $[B, 3, 224, 224]$
- **Conv1 & MaxPool:** Initial $7 \times 7$ convolution (stride 2) + BatchNorm + ReLU + $3 \times 3$ MaxPool (stride 2) $\to [B, 64, 56, 56]$.
- **Residual Stages (Layers 1 to 4):**
  - `Layer 1`: 2 BasicBlocks with 64 channels $\to [B, 64, 56, 56]$
  - `Layer 2`: 2 BasicBlocks with 128 channels (stride 2) $\to [B, 128, 28, 28]$
  - `Layer 3`: 2 BasicBlocks with 256 channels (stride 2) $\to [B, 256, 14, 14]$
  - `Layer 4`: 2 BasicBlocks with 512 channels (stride 2) $\to [B, 512, 7, 7]$
- **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` followed by `Flatten()` produces a 512-dimensional visual feature vector:
  $$\mathbf{f}_v = \text{Flatten}(\text{AdaptiveAvgPool2d}(\text{ResNet18}(x))) \in \mathbb{R}^{512}$$

### Partial Fine-Tuning Strategy
To preserve low-level visual features while adapting high-level representations to wildlife ecology:
- The early layers (`conv1`, `layer1`) remain stable with low learning rate.
- Deep semantic layers (`layer3`, `layer4`) are fine-tuned with a restrained backbone learning rate ($\eta_{\text{backbone}} = 10^{-5}$).

---

## 5. BiLSTM Text Encoder

The text encoder maps variable-length token sequences into fixed-size semantic representations using a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

### Architecture Details
1. **Embedding Layer:**
   $$\mathbf{e}_t = \text{Dropout}(\mathbf{W}_{\text{emb}}[w_t]), \quad \mathbf{W}_{\text{emb}} \in \mathbb{R}^{158 \times 128}$$
   The `<PAD>` index (0) gradients are zeroed during training.
2. **Recurrent Encoding:**
   A single-layer BiLSTM processes the sequence in both forward ($\vec{\mathbf{h}}_t$) and reverse ($\overleftarrow{\mathbf{h}}_t$) temporal directions:
   $$\mathbf{h}_t = [\vec{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{256}, \quad t \in \{1, \dots, 32\}$$
3. **Padding-Aware Masked Mean Pooling:**
   Rather than using the arbitrary final hidden state (which corresponds to meaningless trailing padding tokens), the encoder computes a masked temporal average over valid tokens:
   $$\mathbf{m}_t = \mathbb{I}(w_t \neq \text{PAD}), \quad \mathbf{f}_t = \frac{\sum_{t=1}^{32} \mathbf{h}_t \cdot \mathbf{m}_t}{\sum_{t=1}^{32} \mathbf{m}_t} \in \mathbb{R}^{256}$$

---

## 6. Projection Heads & Shared Embedding Space

Both visual and textual feature extractors feed into independent two-layer projection Multilayer Perceptrons (MLPs):

### Image Projection Head
$$\mathbf{z}_v = \mathbf{W}_2^{(v)} \cdot \text{Dropout}\left(\text{ReLU}\left(\mathbf{W}_1^{(v)} \cdot \mathbf{f}_v + \mathbf{b}_1^{(v)}\right)\right) + \mathbf{b}_2^{(v)}$$
where $\mathbf{W}_1^{(v)} \in \mathbb{R}^{512 \times 512}$ and $\mathbf{W}_2^{(v)} \in \mathbb{R}^{256 \times 512}$.

### Text Projection Head
$$\mathbf{z}_t = \mathbf{W}_2^{(t)} \cdot \text{Dropout}\left(\text{ReLU}\left(\mathbf{W}_1^{(t)} \cdot \mathbf{f}_t + \mathbf{b}_1^{(t)}\right)\right) + \mathbf{b}_2^{(t)}$$
where $\mathbf{W}_1^{(t)} \in \mathbb{R}^{256 \times 256}$ and $\mathbf{W}_2^{(t)} \in \mathbb{R}^{256 \times 256}$.

### Unit $L_2$ Normalization
Embeddings are projected onto the unit hypersphere $\mathbb{S}^{255}$:
$$\mathbf{v} = \frac{\mathbf{z}_v}{\|\mathbf{z}_v\|_2}, \quad \mathbf{t} = \frac{\mathbf{z}_t}{\|\mathbf{z}_t\|_2} \quad \implies \quad \|\mathbf{v}\|_2 = 1.0, \quad \|\mathbf{t}\|_2 = 1.0$$
In this normalized space, cosine similarity is equivalent to the Euclidean inner product:
$$\text{Sim}(\mathbf{v}, \mathbf{t}) = \mathbf{v} \cdot \mathbf{t}^T \in [-1.0, 1.0]$$

---

## 7. Species-Aware Multi-Positive Contrastive Learning

### The False Negative Problem in Standard InfoNCE
In standard instance-level InfoNCE contrastive learning:
$$\mathcal{L}_{\text{InfoNCE}}^{(i)} = -\log \frac{\exp(\mathbf{v}_i \cdot \mathbf{t}_i / \tau)}{\sum_{j=1}^B \exp(\mathbf{v}_i \cdot \mathbf{t}_j / \tau)}$$
Only the single diagonal pair $(i, i)$ is treated as positive. All other batch elements $(i, j)$ ($j \neq i$) are penalized as negatives.

In a wildlife dataset with $K=4$ classes and batch size $B=16$, approximately 4 samples in every batch belong to the same species. When sample $i$ is a tiger and sample $j$ is another tiger, standard InfoNCE actively forces their representations apart! This introduces severe gradient conflict and degrades cross-modal semantic clustering.

### Species-Aware Multi-Positive Formulation
Version 3 implements **Species-Aware Multi-Positive Contrastive Learning** (Supervised Contrastive Multi-Positive InfoNCE):
Let $P(i) = \{j \in \{1, \dots, B\} \mid \text{species}(j) = \text{species}(i)\}$ denote the set of all positive pairs in the batch sharing the same species label:

$$\mathcal{L}_{\text{multi-pos}}^{(v \to t)} = -\frac{1}{B} \sum_{i=1}^B \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{v}_i \cdot \mathbf{t}_p / \tau)}{\sum_{k=1}^B \exp(\mathbf{v}_i \cdot \mathbf{t}_k / \tau)}$$

$$\mathcal{L}_{\text{multi-pos}}^{(t \to v)} = -\frac{1}{B} \sum_{j=1}^B \frac{1}{|P(j)|} \sum_{p \in P(j)} \log \frac{\exp(\mathbf{t}_j \cdot \mathbf{v}_p / \tau)}{\sum_{k=1}^B \exp(\mathbf{t}_j \cdot \mathbf{v}_k / \tau)}$$

$$\mathcal{L}_{\text{contrastive}} = \frac{1}{2} \left(\mathcal{L}_{\text{multi-pos}}^{(v \to t)} + \mathcal{L}_{\text{multi-pos}}^{(t \to v)}\right)$$

This objective encourages all tigers to cluster near all tiger captions while pushing lions, elephants, and deer into well-separated clusters on $\mathbb{S}^{255}$.

---

## 8. Auxiliary Classification Objective

To further stabilize the multimodal latent space and ground embeddings in biological taxonomy, Version 3 introduces auxiliary single-layer linear classification heads:

$$\hat{\mathbf{y}}_v = \mathbf{W}_{\text{cls}}^{(v)} \mathbf{v} + \mathbf{b}_{\text{cls}}^{(v)}, \quad \hat{\mathbf{y}}_t = \mathbf{W}_{\text{cls}}^{(t)} \mathbf{t} + \mathbf{b}_{\text{cls}}^{(t)}, \quad \mathbf{W}_{\text{cls}} \in \mathbb{R}^{4 \times 256}$$

The auxiliary loss is the standard multi-class cross-entropy on both modalities:
$$\mathcal{L}_{\text{aux}} = \frac{1}{2} \left(\text{CE}(\hat{\mathbf{y}}_v, \mathbf{y}) + \text{CE}(\hat{\mathbf{y}}_t, \mathbf{y})\right)$$

### Total Joint Optimization Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{contrastive}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
The auxiliary loss acts as an explicit regularizer preventing representation collapse and accelerating convergence.

---

## 9. Training Configuration & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Optimizer** | AdamW | Decoupled weight decay regularization |
| **Weight Decay** | $10^{-4}$ | Prevents weight explosion in projection heads |
| **Backbone Learning Rate ($\eta_{\text{bb}}$)** | $10^{-5}$ | Conservative fine-tuning rate for ResNet-18 |
| **Heads Learning Rate ($\eta_{\text{new}}$)** | $2 \times 10^{-4}$ | Fast convergence rate for BiLSTM & projection MLPs |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor $0.5$, patience 3 epochs on validation recall |
| **Batch Size ($B$)** | 16 | Balanced mini-batches with stratified species samples |
| **Max Epochs** | 35 | Trained for 34 complete epochs |
| **Initial Temperature ($\tau_0$)** | 0.07 | Scale factor initialized to $\ln(1/0.07) \approx 2.6593$ |
| **Temperature Clamp** | 100.0 | Prevents gradient explosion in similarity scaling |
| **Auxiliary Loss Weight ($\lambda_{\text{aux}}$)** | 0.5 | Balances contrastive and classification gradients |
| **Dropout Probability** | 0.1 | Applied to word embeddings and projection heads |

---

## 10. Early Stopping & Checkpoint Selection

Training dynamics were monitored across all 34 epochs:
- Total loss decreased from **1.750** (Epoch 1) to **0.367** (Epoch 34).
- Contrastive loss decreased from **1.263** to **0.013**.
- Auxiliary image CE loss decreased from **1.392** to **0.912**.
- Auxiliary text CE loss decreased from **1.389** to **1.113**.

### Best Epoch Selection
Model selection was governed by **Validation Species Retrieval Recall@1**:
- Epoch 19 achieved **100.0% Validation Species Retrieval Recall@1** and **100.0% Validation Species Retrieval Recall@5**.
- Epoch 19 weights were captured and frozen as `wildmatch_v3_best.pth` and `wildmatch_v3_dual_encoder.pth`.

---

## 11. Evaluation Methodology

Evaluation was conducted on the held-out test set ($N=30$) across two complementary retrieval paradigms:

### Paradigm A: Species-Level Semantic Retrieval
Evaluates whether a query retrieves any gallery candidate belonging to the correct biological species:
$$\text{SpeciesRecall@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\text{species}(i) \in \{\text{species}(\text{top}_k)\}_{k=1}^K\right)$$

### Paradigm B: Exact Image-Caption Retrieval
Evaluates whether a query retrieves its single exact original pair:
$$\text{ExactRecall@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(i \in \{\text{top}_k\}_{k=1}^K\right)$$

---

## 12. Version 1 vs. Version 2 vs. Version 3 Progression

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ARCHITECTURAL PROGRESSION SUMMARY                                 │
├───────────┬──────────────────────┬──────────────────────┬────────────────────────────────────────┤
│ Version   │ Visual Backbone      │ Loss Formulation     │ Species Retrieval Recall@1             │
├───────────┼──────────────────────┼──────────────────────┼────────────────────────────────────────┤
│ Version 1 │ Scratch 4-layer CNN  │ Random Init / InfoNCE│ ~25.0% (Random Guessing)               │
│ Version 2 │ ResNet-18 (standard) │ Standard InfoNCE     │ ~65.0% (Degraded by False Negatives)   │
│ Version 3 │ ResNet-18 (fine-tune)│ Multi-Pos + Aux CE   │ 93.33% (100% on Text-to-Image)         │
└───────────┴──────────────────────┴──────────────────────┴────────────────────────────────────────┘
```

1. **Version 1 (Untrained Random Baseline):**
   Cosine similarities clustered near zero ($+0.01$ to $+0.03$), demonstrating expected orthogonal geometry on $\mathbb{S}^{255}$.
2. **Version 2 (Standard InfoNCE with ResNet-18):**
   Stronger visual representation, but suffered from false negatives: pushing apart multiple tigers and lions within the same mini-batch.
3. **Version 3 (Delivered Solution):**
   Species-aware multi-positive contrastive learning allowed all intra-species pairs to attract, while auxiliary cross-entropy heads grounded the embeddings. Achieved 93.33% average species Recall@1.

---

## 13. Final Experimental Results

The following verified metrics were obtained on the held-out test split of 30 samples:

| Metric | Result | Target / Interpretation |
|---|---:|---|
| **Text-to-Image Species Recall@1** | **100.00%** | All 30 text queries retrieve the correct species at Rank 1 |
| **Text-to-Image Species Recall@5** | **100.00%** | All 30 text queries retrieve the correct species in Top-5 |
| **Image-to-Text Species Recall@1** | **86.67%** | 26 of 30 image queries retrieve matching species text at Rank 1 |
| **Image-to-Text Species Recall@5** | **90.00%** | 27 of 30 image queries retrieve matching species text in Top-5 |
| **Image Classifier Accuracy** | **86.67%** | Auxiliary head correctly predicts species for 26/30 images |
| **Text Classifier Accuracy** | **100.00%** | Auxiliary head correctly predicts species for 30/30 captions |
| **Average Species Recall@1** | **93.33%** | Mean bimodal species retrieval accuracy |
| **Average Species Recall@5** | **95.00%** | Mean bimodal Top-5 species accuracy |
| **Text-to-Image Exact Recall@1** | **16.67%** | Exact pair retrieved at Rank 1 (5/30) |
| **Text-to-Image Exact Recall@5** | **66.67%** | Exact pair retrieved in Top-5 (20/30) |
| **Image-to-Text Exact Recall@1** | **16.67%** | Exact pair retrieved at Rank 1 (5/30) |
| **Image-to-Text Exact Recall@5** | **53.33%** | Exact pair retrieved in Top-5 (16/30) |
| **Average Exact Recall@1** | **16.67%** | Mean bimodal Top-1 exact retrieval |
| **Average Exact Recall@5** | **60.00%** | Mean bimodal Top-5 exact retrieval |

---

## 14. Limitations

1. **Species Scope:** Constrained to the 4 trained focal species (*Asian Elephant, Chital, Lion, Tiger*). Unseen species will project into the nearest semantic subspace.
2. **Vocabulary Breadth:** 158 words. Rare natural language terms outside this vocabulary map to `<UNK>`.
3. **Exact-Instance Discrimination:** Because the loss explicitly rewards grouping images of the same species together, fine-grained distinction between individual animals (e.g. distinguishing Tiger A from Tiger B) is subordinate to species-level clustering.

---

## 15. Downstream Integration Instructions for Member 4

1. **Direct Checkpoint Loading:**
   Member 4 should load models using `models.inference.load_model("models/wildmatch_v3_dual_encoder.pth")`.
2. **Inference Pipeline:**
   Use `models.inference.preprocess_image` and `models.inference.tokenize_caption`.
3. **Auxiliary Tags in UI:**
   Member 4's Gradio app should display both the retrieval similarity score and the predicted species tag from `models.inference.load_species_heads`.
4. **Offline Operation:**
   The entire inference pipeline operates with zero external network calls or remote model downloads.

