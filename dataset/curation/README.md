# WildMatch AI — Visual Curation Guide

This directory contains the tools and outputs for manually reviewing and curating the raw dataset before caption generation.

---

## 1. Directory Structure

```text
dataset/
├── metadata.csv                      # Source of truth (raw dataset)
├── raw/                              # Original untouched images
│   ├── tiger/
│   ├── lion/
│   ├── elephant/
│   └── chital/
└── curation/
    ├── README.md                     # This guide
    ├── curation_status.csv           # Manual review decisions (KEEP / REJECT / REVIEW)
    └── contact_sheets/               # Visual grids for each species
        ├── tiger_contact_sheet.jpg
        ├── lion_contact_sheet.jpg
        ├── elephant_contact_sheet.jpg
        └── chital_contact_sheet.jpg
```

---

## 2. Review Workflow

### Step 1: Inspect Contact Sheets
Open the contact sheets in `dataset/curation/contact_sheets/` using any standard image viewer:
- `tiger_contact_sheet.jpg` (25 images)
- `lion_contact_sheet.jpg` (10 images)
- `elephant_contact_sheet.jpg` (10 images)
- `chital_contact_sheet.jpg` (10 images)

Each cell shows the image thumbnail, original filename, photo ID, and original dimensions.

### Step 2: Set Curation Decisions
Open `dataset/curation/curation_status.csv` in Excel, a CSV editor, or any text editor.

For each row, update:
- **`curation_status`**:
  - `KEEP` — High quality, clear subject, appropriate for captioning and retrieval.
  - `REJECT` — Heavily occluded, blurry, misidentified species, extreme close-up/crop, or unnatural context.
  - `REVIEW` — Default pending state. *(The pipeline will refuse finalization until all rows are decided).*
- **`rejection_reason`** *(Optional for REJECT)*:
  - e.g., `heavy_occlusion`, `motion_blur`, `unnatural_setting`, `extreme_crop`.
- **`reviewer_notes`** *(Optional)*:
  - Any notes regarding lighting, sex/age, or framing.

### Step 3: Regenerate Sheets (Optional)
If new images are added or you want to refresh the sheets, simply rerun:
```powershell
python preprocessing/curate_dataset.py --generate-sheets
```
Existing decisions, rejection reasons, and reviewer notes in `curation_status.csv` are **preserved**.

### Step 4: Finalize the Curated Dataset
Once **every** record in `curation_status.csv` is set to either `KEEP` or `REJECT` (0 records remaining in `REVIEW`):

```powershell
python preprocessing/curate_dataset.py --finalize
```

This operation will:
1. Copy all `KEEP` images into `dataset/curated/{species}/`.
2. Generate `dataset/curated/metadata.csv` preserving all original metadata fields.
3. Leave raw images in `dataset/raw/` completely untouched.
4. Exclude all `REJECT` images from the curated dataset.
