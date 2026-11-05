# WildMatch AI - Dataset Splits Documentation

## Overview
This directory contains the reproducible, stratified train/validation/test splits for the 200-record curated WildMatch AI dataset.

## Split Configuration
- **Random Seed**: `42` (via `numpy.random.RandomState(42)`)
- **Source Dataset**: `dataset/captions/processed_captions.csv`
- **Stratification Key**: `species`
- **Split Proportions**: 70% Train, 15% Validation, 15% Test

## Distribution Table

| Species | Train | Validation | Test | Total |
|---|:---:|:---:|:---:|:---:|
| **Tiger** | 35 | 7 | 8 | 50 |
| **Lion** | 35 | 8 | 7 | 50 |
| **Asian Elephant** | 35 | 7 | 8 | 50 |
| **Chital** | 35 | 8 | 7 | 50 |
| **Total** | **140** | **30** | **30** | **200** |

## Files
- `train.csv`: 140 training records (35 per species) with cleaned captions and tokens.
- `validation.csv`: 30 validation records (7 Tiger, 8 Lion, 7 Elephant, 8 Chital).
- `test.csv`: 30 test records (8 Tiger, 7 Lion, 8 Elephant, 7 Chital).
- `split_manifest.csv`: Mapping of `photo_id`, `image_id`, `species`, `split`, and `image_path`.

## Data Leakage & Duplicate Audit
- **Exact File Duplicates (MD5)**: 0
- **Near-Duplicate Check (dHash <= 8)**: Exactly 1 pair identified: `690211270` and `690211361` (Tiger).
  Both images are strictly assigned to the **Train** split, preventing any cross-split data leakage.
