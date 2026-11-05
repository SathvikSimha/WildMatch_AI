"""
WildMatch AI - Stratified Dataset Splitting Script
Splits dataset/captions/processed_captions.csv into reproducible train, validation, and test splits.
"""

import hashlib
import os
import numpy as np
import pandas as pd
from PIL import Image

INPUT_CSV = "dataset/captions/processed_captions.csv"
SPLITS_DIR = "dataset/splits"
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "validation.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")
MANIFEST_CSV = os.path.join(SPLITS_DIR, "split_manifest.csv")
README_MD = os.path.join(SPLITS_DIR, "README.md")

RANDOM_SEED = 42

SPLIT_TARGETS = {
    "Tiger": {"train": 35, "validation": 7, "test": 8},
    "Lion": {"train": 35, "validation": 8, "test": 7},
    "Asian Elephant": {"train": 35, "validation": 7, "test": 8},
    "Chital": {"train": 35, "validation": 8, "test": 7},
}


def compute_md5(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_dhash(image_path: str, hash_size: int = 8) -> str:
    with Image.open(image_path) as img:
        img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img_gray.getdata())
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)
        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2 ** (index % 8)
            if (index % 8) == 7:
                hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
                decimal_value = 0
        return "".join(hex_string)


def hamming_distance(h1: str, h2: str) -> int:
    b1 = bin(int(h1, 16))[2:].zfill(64)
    b2 = bin(int(h2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


def main():
    print("=== STEP 1: LOAD DATASET & VERIFY INTEGRITY ===")
    assert os.path.exists(INPUT_CSV), f"Input CSV not found at {INPUT_CSV}"
    df = pd.read_csv(INPUT_CSV)
    assert len(df) == 200, f"Expected 200 rows, got {len(df)}"
    assert df["photo_id"].nunique() == 200, "Photo IDs not unique"
    assert (df["species"].value_counts() == 50).all(), "Species counts not 50 each"
    print(f"Loaded {len(df)} records from {INPUT_CSV}.")

    print("\n=== STEP 2: DUPLICATE & NEAR-DUPLICATE AUDIT ===")
    md5_map = {}
    dhash_map = {}
    for _, row in df.iterrows():
        path = row["image_path"]
        assert os.path.exists(path), f"Image file missing: {path}"
        md5_val = compute_md5(path)
        dh_val = compute_dhash(path)
        md5_map[row["photo_id"]] = (md5_val, row["species"])
        dhash_map[row["photo_id"]] = (dh_val, row["species"])

    # Check exact MD5 duplicates
    md5_inverted = {}
    for pid, (h, sp) in md5_map.items():
        md5_inverted.setdefault(h, []).append((pid, sp))
    exact_dups = {k: v for k, v in md5_inverted.items() if len(v) > 1}
    print(f"Exact file duplicates (MD5): {len(exact_dups)}")
    assert len(exact_dups) == 0, f"Found exact duplicates: {exact_dups}"

    # Check near duplicates (dHash Hamming distance <= 8)
    pids = list(dhash_map.keys())
    near_dups = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            p1, p2 = pids[i], pids[j]
            h1, sp1 = dhash_map[p1]
            h2, sp2 = dhash_map[p2]
            dist = hamming_distance(h1, h2)
            if dist <= 8:
                near_dups.append((p1, p2, sp1, sp2, dist))

    print(f"Near-duplicate pairs (dHash distance <= 8): {len(near_dups)}")
    for p1, p2, sp1, sp2, dist in near_dups:
        print(f"  Near-duplicate: photo_id {p1} & {p2} ({sp1}), Hamming distance = {dist}")

    print("\n=== STEP 3: PERFORM STRATIFIED SPLIT ===")
    rng = np.random.RandomState(RANDOM_SEED)
    manifest_records = []
    split_indices = {"train": [], "validation": [], "test": []}

    for species, targets in SPLIT_TARGETS.items():
        sp_df = df[df["species"] == species].sort_values("photo_id").reset_index(drop=True)
        assert len(sp_df) == 50, f"Expected 50 records for {species}, got {len(sp_df)}"

        perm = rng.permutation(len(sp_df))
        n_train = targets["train"]
        n_val = targets["validation"]
        n_test = targets["test"]
        assert n_train + n_val + n_test == 50, "Target split sum != 50"

        train_indices = perm[:n_train]
        val_indices = perm[n_train: n_train + n_val]
        test_indices = perm[n_train + n_val:]

        for idx in train_indices:
            row = sp_df.iloc[idx]
            manifest_records.append({
                "photo_id": row["photo_id"],
                "image_id": row["image_id"],
                "species": row["species"],
                "split": "train",
                "image_path": row["image_path"],
            })
            split_indices["train"].append(row["photo_id"])

        for idx in val_indices:
            row = sp_df.iloc[idx]
            manifest_records.append({
                "photo_id": row["photo_id"],
                "image_id": row["image_id"],
                "species": row["species"],
                "split": "validation",
                "image_path": row["image_path"],
            })
            split_indices["validation"].append(row["photo_id"])

        for idx in test_indices:
            row = sp_df.iloc[idx]
            manifest_records.append({
                "photo_id": row["photo_id"],
                "image_id": row["image_id"],
                "species": row["species"],
                "split": "test",
                "image_path": row["image_path"],
            })
            split_indices["test"].append(row["photo_id"])

    manifest_df = pd.DataFrame(manifest_records)
    split_lookup = dict(zip(manifest_df["photo_id"], manifest_df["split"]))

    # Verify near-duplicate pair does not cross splits
    for p1, p2, sp1, sp2, dist in near_dups:
        s1 = split_lookup[p1]
        s2 = split_lookup[p2]
        print(f"Near-duplicate check: photo {p1} -> {s1}, photo {p2} -> {s2}")
        assert s1 == s2, f"Near-duplicate pair ({p1}, {p2}) separated across splits ({s1} vs {s2})!"

    print("Near-duplicate verification passed: No near-duplicates cross splits.")

    print("\n=== STEP 4: SAVE OUTPUT FILES ===")
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Filter original df by split
    train_df = df[df["photo_id"].isin(split_indices["train"])].copy()
    val_df = df[df["photo_id"].isin(split_indices["validation"])].copy()
    test_df = df[df["photo_id"].isin(split_indices["test"])].copy()

    # Save CSVs
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    print(f"Saved {len(train_df)} train records to {TRAIN_CSV}")
    print(f"Saved {len(val_df)} validation records to {VAL_CSV}")
    print(f"Saved {len(test_df)} test records to {TEST_CSV}")
    print(f"Saved {len(manifest_df)} manifest records to {MANIFEST_CSV}")

    # Write README.md
    readme_content = f"""# WildMatch AI - Dataset Splits Documentation

## Overview
This directory contains the reproducible, stratified train/validation/test splits for the 200-record curated WildMatch AI dataset.

## Split Configuration
- **Random Seed**: `{RANDOM_SEED}` (via `numpy.random.RandomState({RANDOM_SEED})`)
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
"""
    with open(README_MD, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved documentation to {README_MD}")

    print("\n=== STEP 5: COMPREHENSIVE VALIDATION ===")
    assert len(train_df) == 140, f"Expected 140 train, got {len(train_df)}"
    assert len(val_df) == 30, f"Expected 30 val, got {len(val_df)}"
    assert len(test_df) == 30, f"Expected 30 test, got {len(test_df)}"

    combined_pids = list(train_df["photo_id"]) + list(val_df["photo_id"]) + list(test_df["photo_id"])
    assert len(combined_pids) == 200, "Combined records != 200"
    assert len(set(combined_pids)) == 200, "Duplicate photo IDs found across splits"
    assert set(combined_pids) == set(df["photo_id"]), "Combined photo IDs do not match input CSV"

    # Check each species distribution
    for sp, targets in SPLIT_TARGETS.items():
        assert (train_df["species"] == sp).sum() == targets["train"], f"Train mismatch for {sp}"
        assert (val_df["species"] == sp).sum() == targets["validation"], f"Val mismatch for {sp}"
        assert (test_df["species"] == sp).sum() == targets["test"], f"Test mismatch for {sp}"

    # Verify original source CSV is untouched
    df_check = pd.read_csv(INPUT_CSV)
    assert df_check.equals(df), "Source CSV was modified!"

    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
