# WildMatch AI --- Dataset Collection & Preprocessing

**Module:** Member 1 --- Dataset Collection & Preprocessing\
**Project:** WildMatch AI --- Fine-Grained Wildlife Image--Text Semantic
Matching Using Contrastive Deep Learning\
**Dataset status:** Prepared and validated\
**Final curated dataset:** 200 image--caption pairs

------------------------------------------------------------------------

## 1. Module overview

This module prepares the wildlife image--text data used by WildMatch AI.
Its workflow covers:

1.  Dataset research and selection
2.  Wildlife image collection
3.  Image curation and metadata preservation
4.  Caption generation and manual review
5.  Image preprocessing
6.  Text cleaning and tokenization
7.  Train/validation/test splitting
8.  Training-only vocabulary construction
9.  PyTorch Dataset and DataLoader implementation
10. Validation and handoff to downstream model components

The output is a reusable data pipeline supplying paired image tensors,
caption-token tensors, species labels, and identifiers.

## 2. Dataset summary

  Species             Images      Share
  ---------------- --------- ----------
  Tiger                   50        25%
  Lion                    50        25%
  Asian Elephant          50        25%
  Chital                  50        25%
  **Total**          **200**   **100%**

Each image has one manually reviewed final caption. The original
BLIP-generated caption is retained separately.

### Source and collection

-   iNaturalist was used as the source of wildlife photographs.
-   Collection targeted the four selected species rather than
    downloading the full iNaturalist dataset.
-   Photo-level licence filtering accepted **CC-BY** and **CC-BY-SA**.
-   Source attribution and licence metadata were preserved.
-   Research-grade observations were used during collection.
-   The collection workflow used a conservative one-second API request
    interval with retry/backoff handling.

The raw collection reached 280 images. After visual curation, 200 images
were retained: 50 per species. The expansion curation added 164 approved
images to the original 36 approved images. Rejected candidates and
reasons were recorded in curation status files. Approved images were
copied into the curated dataset; raw originals were retained.

### Final curated licence distribution

  Licence        Images      Share
  ----------- --------- ----------
  CC-BY             182        91%
  CC-BY-SA           18         9%
  **Total**     **200**   **100%**

Preserve attribution and licence metadata when redistributing or
publishing the images. Apply the conditions of each image's licence.

## 3. Dataset setup

Image binary files are hosted separately on Google Drive to keep the GitHub repository lightweight (< 1 MB) and ensure fast cloning. All captions, processed captions, vocabulary, curation metadata, and train/validation/test split CSVs are tracked directly in GitHub and do not need to be downloaded separately.

### Google Drive download link

[**Download WildMatch AI Dataset Archives on Google Drive**](https://drive.google.com/drive/folders/13XJRsQHok8HqZi3kYecnlEcWoTWI4gVY?usp=sharing)

The shared Google Drive folder provides two ZIP archives:
- **`wildmatch_curated_dataset.zip`** (~70 MB) — Contains all 200 curated, verified wildlife images (50 per species across Tiger, Lion, Asian Elephant, and Chital). **Required for model development, training, and DataLoader verification.**
- **`wildmatch_raw_dataset.zip`** (~100 MB) — Contains all 280 original collected images from iNaturalist before curation. **Optional for model training**; provided for full data lineage, archival completeness, and re-running raw curation pipelines.

### Installation instructions

Clone the repository and install the project dependencies:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd WildMatch_AI
pip install -r requirements.txt
```

### Dataset extraction instructions

1. Download **`wildmatch_curated_dataset.zip`** from the [Google Drive link](https://drive.google.com/drive/folders/13XJRsQHok8HqZi3kYecnlEcWoTWI4gVY?usp=sharing).
2. Extract the archive into the repository's `dataset/` directory.
3. Verify that the extracted curated images reside in `dataset/curated/` organized by lowercase species subdirectories:
   - `dataset/curated/tiger/` (50 images)
   - `dataset/curated/lion/` (50 images)
   - `dataset/curated/elephant/` (50 images)
   - `dataset/curated/chital/` (50 images)
4. *(Optional)* Download and extract **`wildmatch_raw_dataset.zip`** into `dataset/` to obtain `dataset/raw/` (`tiger/`, `lion/`, `elephant/`, `chital/`).

> [!IMPORTANT]
> **Preserve Folder Structure & Filenames:** All CSV files (`train.csv`, `validation.csv`, `test.csv`, `split_manifest.csv`, `processed_captions.csv`, `pilot_captions.csv`) use project-root-relative image paths (e.g. `dataset/curated/tiger/tiger_759553570.jpg`). Teammates must preserve the exact directory nesting and original image filenames so that all paths resolve seamlessly across Windows, macOS, and Linux without code modifications.

### Expected directory structure after extraction

```text
WildMatch_AI/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset/
│   ├── dataloader.py                     # DataLoader factory
│   ├── wildlife_dataset.py               # PyTorch Dataset
│   ├── vocabulary.json                   # Training-only vocabulary (158 tokens)
│   ├── metadata.csv                      # Full raw collection metadata (280 records)
│   ├── collection_report.csv             # Ingestion summary report
│   ├── captions/
│   │   ├── pilot_captions.csv             # 200 BLIP and reviewed captions
│   │   ├── processed_captions.csv         # Cleaned captions and tokens
│   │   └── review_contact_sheets/
│   │       └── caption_review_index.csv  # Contact sheet review index
│   ├── curation/
│   │   ├── curation_status.csv           # Initial curation decisions (55 records)
│   │   ├── expansion_curation_status.csv # Expansion curation decisions (225 records)
│   │   └── README.md                     # Visual curation documentation
│   ├── splits/
│   │   ├── train.csv                     # 140 training records (35 per species)
│   │   ├── validation.csv                # 30 validation records (7-8 per species)
│   │   ├── test.csv                      # 30 test records (7-8 per species)
│   │   ├── split_manifest.csv            # All 200 record split assignments
│   │   └── README.md                     # Split configuration & leakage audit
│   ├── curated/                          # Extracted from wildmatch_curated_dataset.zip
│   │   ├── metadata.csv                  # Curated metadata (200 records)
│   │   ├── tiger/                        # 50 curated images
│   │   ├── lion/                         # 50 curated images
│   │   ├── elephant/                     # 50 curated images
│   │   └── chital/                       # 50 curated images
│   └── raw/                              # (Optional) Extracted from wildmatch_raw_dataset.zip
│       ├── tiger/                        # 70 raw images
│       ├── lion/                         # 70 raw images
│       ├── elephant/                     # 70 raw images
│       └── chital/                       # 70 raw images
├── preprocessing/
│   ├── collect_inaturalist.py            # Automated iNaturalist data collection
│   ├── curate_dataset.py                 # Visual curation tool & status manager
│   ├── finalize_expansion_curated.py     # Curated dataset finalization & sync
│   ├── generate_captions.py              # BLIP caption generation module
│   ├── generate_expansion_caption_contact_sheets.py # Caption review contact sheets
│   ├── generate_expansion_contact_sheets.py         # Visual curation sheets
│   ├── generate_review_contact_sheets.py            # Pilot review sheets
│   ├── image_preprocessing.py            # Aspect-preserving image transforms
│   └── text_preprocessing.py             # Caption cleaning & word tokenization
└── scripts/
    ├── apply_consolidated_captions.py    # Consolidated caption updater
    ├── build_vocabulary.py               # Training-only vocabulary builder
    ├── split_dataset.py                  # Stratified train/val/test splitter
    └── validate_dataloaders.py           # 15-point DataLoader verification suite
```

### Dataset validation

After extracting the dataset, run the validation script from the repository root to verify that all images, captions, vocabulary, and DataLoaders are configured correctly:

```bash
python scripts/validate_dataloaders.py
```

A successful run verifies all 15 integrity checks, confirming:
1. Dataset lengths: 140 train, 30 validation, 30 test.
2. Image batch shape `[B, 3, 224, 224]`.
3. Caption batch shape `[B, 32]`.
4. Image dtype `float32`.
5. Caption dtype `torch.long`.
6. Image values strictly within `[0.0, 1.0]`.
7. Exactly three RGB channels per image.
8. Caption indices within vocabulary range `[0, 157]`.
9. Sequence padding strictly uses index 0 (`<PAD>`) following `<EOS>`.
10. Unknown words map strictly to index 1 (`<UNK>`).
11. Species labels and integer indices are 100% valid and consistent.
12. Photo IDs and Image IDs are preserved.
13. All 200 image files load without corruption or NaNs.
14. Validation and test loaders preserve sequential order (no shuffle).
15. Training loader shuffles while retaining all 140 elements.

## 4. Repository structure

Principal files and directories used or created by this workflow:

``` text
WildMatch_AI/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset/
│   ├── raw/                              # Original collected images (optional)
│   ├── curated/                          # Curated images (required from Google Drive)
│   │   ├── tiger/                        # 50 curated images
│   │   ├── lion/                         # 50 curated images
│   │   ├── elephant/                     # 50 curated images
│   │   ├── chital/                       # 50 curated images
│   │   └── metadata.csv                  # Curated image metadata
│   ├── captions/
│   │   ├── pilot_captions.csv             # 200 BLIP and reviewed captions
│   │   └── processed_captions.csv         # Cleaned captions and tokens
│   ├── splits/
│   │   ├── train.csv                     # 140 records
│   │   ├── validation.csv                # 30 records
│   │   ├── test.csv                      # 30 records
│   │   ├── split_manifest.csv            # Assignment for all 200 records
│   │   └── README.md                     # Split configuration/report
│   ├── vocabulary.json                   # Training-only vocabulary
│   ├── wildlife_dataset.py               # PyTorch Dataset
│   └── dataloader.py                     # DataLoader factory
├── preprocessing/
│   ├── curate_dataset.py
│   ├── image_preprocessing.py
│   └── text_preprocessing.py
└── scripts/
    ├── split_dataset.py
    ├── build_vocabulary.py
    └── validate_dataloaders.py
```

This is the principal workflow structure, not necessarily an exhaustive
listing of every intermediate script, contact sheet, backup, or report
in the repository.

## 5. Image collection and curation

### Collection constraints

-   Target species: Tiger, Lion, Asian Elephant, Chital
-   Accepted photo licences: CC-BY and CC-BY-SA
-   Photo-level licence filtering
-   Attribution and licence metadata retained
-   Research-grade observations used during collection
-   Conservative API request interval and retry/backoff behaviour

### Curation

Contact sheets supported visual inspection. Decisions were recorded in
curation status CSV files, including KEEP/REJECT status and rejection
reasons.

Final curated dataset checks reported: - 200 unique photo IDs - Exactly
50 images per species - 200 readable, decodable RGB JPEG files -
Original raw images and metadata retained

Four initially questionable images were inspected at full resolution and
retained after their species identity was confirmed.

## 6. Caption generation and review

Captions were generated using:

-   Model: `Salesforce/blip-image-captioning-base`
-   Library: Hugging Face Transformers
-   Generation settings: `num_beams=5`, `max_new_tokens=50`,
    `early_stopping=True`
-   Execution: CPU

The caption CSV retains: - `blip_caption`: original BLIP-generated
output - `final_caption`: manually reviewed caption

BLIP outputs were not overwritten. Captions were corrected when the
generated text misidentified an animal or described details unsupported
by the image.

Final caption dataset: - 200 records - 200 populated reviewed final
captions - 50 records per species

## 7. Image preprocessing

Implementation: `preprocessing/image_preprocessing.py`

### Standard image specification

  Property                                 Standard
  ---------------------------------------- -------------------------
  Dimensions                               224 × 224
  Format                                   JPEG (`.jpg` / `.jpeg`)
  Colour                                   RGB, 3 channels
  Tensor layout                            `[3, 224, 224]`
  Batch layout                             `[B, 3, 224, 224]`
  Data type                                `float32`
  Pixel range before model normalization   `[0, 1]`

### Processing steps

1.  Safely open and decode the image.
2.  Apply EXIF orientation correction.
3.  Convert to RGB.
4.  Resize with aspect ratio preserved.
5.  Symmetrically pad to 224 × 224; do not stretch the image.
6.  Convert HWC image data to a channels-first PyTorch tensor.
7.  Return `float32` values in `[0, 1]`.

Original images are retained. Resizing is performed during loading
rather than by replacing source files with resized copies.

### Model-specific normalization

The shared Dataset returns standardized image tensors in `[0, 1]`.
CNN-specific normalization and CLIP-specific preprocessing remain
separate. The CLIP path must use CLIP's own image processor and
tokenizer.

## 8. Text preprocessing

Implementation: `preprocessing/text_preprocessing.py`

**Input:** `dataset/captions/pilot_captions.csv`\
**Output:** `dataset/captions/processed_captions.csv`

The source caption CSV was preserved.

For each record, the workflow: 1. Reads the reviewed `final_caption`. 2.
Converts text to lowercase. 3. Removes punctuation using the existing
cleaning implementation. 4. Normalizes whitespace. 5. Tokenizes using
whitespace-based word tokenization. 6. Retains original captions and
metadata alongside processed text.

The processed CSV includes `cleaned_caption` and `tokens` in addition to
the source fields. Token lists are stored as JSON-encoded lists in the
CSV.

### Sequence settings

-   Maximum sequence length: 32 tokens
-   Special tokens: PAD, UNK, BOS, EOS
-   Vocabulary fitting deferred until the training split was isolated

### Full-dataset preprocessing validation

  Metric                           Result
  ------------------------------ --------
  Processed records                   200
  Unique photo IDs                    200
  Missing final captions                0
  Missing cleaned captions              0
  Tokenization successes              200
  Tokenization failures                 0
  Minimum observed token count          5
  Maximum observed token count         15
  Average observed token count       8.34

All observed caption lengths were within the 32-token limit.

## 9. Train, validation, and test splits

Implementation: `scripts/split_dataset.py`

### Configuration

-   Random seed: `42`
-   Random generator: `numpy.random.RandomState(42)`
-   Stratification field: `species`
-   Overall proportions: 70% train, 15% validation, 15% test

### Split distribution

  Species              Train   Validation     Test     Total
  ---------------- --------- ------------ -------- ---------
  Tiger                   35            7        8        50
  Lion                    35            8        7        50
  Asian Elephant          35            7        8        50
  Chital                  35            8        7        50
  **Total**          **140**       **30**   **30**   **200**

### Generated files

-   `dataset/splits/train.csv`
-   `dataset/splits/validation.csv`
-   `dataset/splits/test.csv`
-   `dataset/splits/split_manifest.csv`
-   `dataset/splits/README.md`

The manifest records the split assignment for each image, including
identifiers and species.

### Duplicate and leakage audit

The split workflow reported: - 0 exact byte-level duplicate files (MD5
check). - 19,900 image pairs compared using difference hash (dHash). -
One near-duplicate pair detected at Hamming distance 2: -
`tiger_690211270.jpg` --- "A tiger resting in shallow water near the
shore." - `tiger_690211361.jpg` --- "A tiger resting in shallow water
near a stream." - Both images were assigned to the training split. - No
detected near-duplicate pair crossed between splits.

The pair remains in the dataset and is documented. It may represent
redundant training content, even though the reported audit found no
cross-split leakage from this pair.

## 10. Training-only vocabulary

Implementation: `scripts/build_vocabulary.py`\
Output: `dataset/vocabulary.json`

Vocabulary was fitted using only `dataset/splits/train.csv`. Validation
and test captions were not used during fitting.

Configuration: - `min_freq=1` - `max_seq_length=32`

### Special-token indices

  Token       Index Meaning
  --------- ------- -------------------------
  `<PAD>`         0 Padding
  `<UNK>`         1 Out-of-vocabulary token
  `<BOS>`         2 Beginning of sequence
  `<EOS>`         3 End of sequence

### Vocabulary statistics

  Metric                       Result
  -------------------------- --------
  Unique training words           154
  Special tokens                    4
  Total vocabulary entries        158
  Index range                  0--157

### Held-out OOV analysis

  Split            Total tokens     In-vocabulary   Mapped to `<UNK>`
  -------------- -------------- ----------------- -------------------
  Validation                245       235 (95.9%)           10 (4.1%)
  Test                      254       245 (96.5%)            9 (3.5%)
  **Combined**          **499**   **480 (96.2%)**       **19 (3.8%)**

Unseen validation and test words map to `<UNK>` at index 1. JSON
serialization and reload validation passed, including token-to-index and
index-to-token consistency.

## 11. PyTorch Dataset and DataLoader

Implementation files: - `dataset/wildlife_dataset.py` -
`dataset/dataloader.py` - Validation script:
`scripts/validate_dataloaders.py`

### Dataset output

Each item returns image data, caption tokens, species information, and
identifiers.

  Field           Type / Shape
  --------------- --------------------------------------------
  `image`         `torch.float32`, `[3, 224, 224]`
  `caption`       `torch.long`, `[32]`
  `species`       String label
  `species_idx`   `torch.long` class index
  `photo_id`      `torch.long` in the current implementation
  `image_id`      String identifier

Species-index mapping:

  Species            Index
  ---------------- -------
  Tiger                  0
  Lion                   1
  Asian Elephant         2
  Chital                 3

### DataLoader configuration

-   Batch size: 16
-   Training shuffle: enabled
-   Validation shuffle: disabled
-   Test shuffle: disabled
-   `num_workers=0` initially for Windows compatibility
-   `pin_memory` enabled only when CUDA is available

### Usage example

``` python
import torch
from dataset.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    splits_dir="dataset/splits",
    vocab="dataset/vocabulary.json",
    batch_size=16,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

batch = next(iter(train_loader))

images = batch["image"]              # [B, 3, 224, 224]
captions = batch["caption"]          # [B, 32]
species = batch["species"]
species_idx = batch["species_idx"]
photo_ids = batch["photo_id"]
image_ids = batch["image_id"]
```

### DataLoader validation

All 15 reported validation checks passed: 1. Dataset lengths: 140 train,
30 validation, 30 test. 2. Image batch shape `[B, 3, 224, 224]`. 3.
Caption batch shape `[B, 32]`. 4. Image dtype `float32`. 5. Caption
dtype `torch.long`. 6. Image values within `[0, 1]`. 7. Exactly three
RGB channels. 8. Caption indices within vocabulary range. 9. Padding
uses index 0. 10. Unknown words map to index 1. 11. Species labels and
indices are consistent. 12. Photo IDs and image IDs are preserved. 13.
All 200 image files load without reported corruption or NaNs. 14.
Validation and test loaders preserve sequential order. 15. Training
loader shuffles while retaining all 140 items.

## 12. Validation summary

  Stage                                          Reported result
  ---------------------------------------------- -------------------------------------
  Curated image dataset                          200 valid RGB JPEG images
  Caption dataset                                200 reviewed final captions
  Text preprocessing                             200/200 successful
  Species balance                                50 per species
  Dataset splitting                              140 train / 30 validation / 30 test
  Unique photo IDs across splits                 200
  Cross-split ID overlap                         0
  Exact duplicate files                          0
  Detected near-duplicate pair crossing splits   0
  Training vocabulary                            158 entries, training split only
  DataLoader validation                          15/15 checks passed

These results summarize the reported validation runs. Re-run the
relevant validation scripts after changing source data, preprocessing
code, split assignments, or loader behaviour.

## 13. Handoff contract for model development

The dataset pipeline supplies paired image and text inputs to downstream
components.

### Custom CNN and BiLSTM

-   Image tensor: `[B, 3, 224, 224]`, `float32`, values `[0, 1]`.
-   Caption tensor: `[B, 32]`, `torch.long`.
-   Vocabulary size: 158.
-   Special-token indices: PAD 0, UNK 1, BOS 2, EOS 3.
-   Downstream code should confirm expected image normalization,
    embedding dimensions, padding behaviour, and sequence conventions.

### CLIP

-   Use CLIP's own image processor and text tokenizer.
-   Do not use the custom 32-token word-level vocabulary as a substitute
    for CLIP tokenization.
-   Keep CLIP-specific normalization separate from the shared `[0, 1]`
    image tensor.

### Integration checks still required

Before project-level closure, downstream members should confirm: 1.
Their model accepts the supplied image and text interfaces. 2.
Normalization and tokenization expectations are explicit. 3.
Image-caption pairing is preserved through embedding and
contrastive-loss computation. 4. Identifiers are retained for
traceability and not inadvertently used as model features.

**Implementation note:** `photo_id` is returned as `torch.long`. Confirm
that downstream code accepts this type. Retain the original identifier
for traceability.

## 14. Limitations and considerations

-   The dataset contains 200 images across four species. It supports a
    project demonstrator, but is small for broad claims about wildlife
    image--text retrieval generalization.
-   BLIP-generated captions were manually reviewed, but remain concise
    descriptions rather than exhaustive annotations.
-   One near-duplicate Tiger image pair was detected in the training
    split.
-   The vocabulary is fitted on only 140 training captions; the held-out
    OOV rate is reported above.
-   CLIP preprocessing and tokenization must remain separate from the
    custom CNN/BiLSTM data path.
-   Image attribution and licence metadata should be retained in any
    redistributed dataset.

## 15. Member 1 completion status

The Member 1 dataset collection and preprocessing implementation has
been completed and validated according to the workflow reports.

### Completed

-   Dataset research and collection
-   Image curation
-   Caption generation and review
-   Image preprocessing
-   Text preprocessing
-   Train/validation/test splitting
-   Training-only vocabulary construction
-   PyTorch Dataset and DataLoader implementation
-   Validation across all 200 records

### Remaining for project-level closure

-   Confirm downstream CNN/BiLSTM/contrastive-learning interface
    compatibility.
-   Integrate the module with the other members' code.
-   Run an end-to-end integration test.
-   Keep this README and the split manifest with the project repository.

------------------------------------------------------------------------

**Module:** Member 1 --- Dataset Collection & Preprocessing\
**Project:** WildMatch AI
