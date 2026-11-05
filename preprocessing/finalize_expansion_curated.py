"""
WildMatch AI - Finalize Expanded Curated Dataset

This script handles:
1. Validating pre-conditions:
   - 164 newly approved KEEP images in `dataset/curation/expansion_curation_status.csv`
     (Tiger: 33, Lion: 40, Asian Elephant: 46, Chital: 45).
   - 36 existing curated images in `dataset/curated/metadata.csv`
     (Tiger: 17, Lion: 10, Asian Elephant: 4, Chital: 5).
2. Copying (never moving) all 164 newly approved images from `dataset/raw/` into `dataset/curated/`.
3. Creating a safety backup of existing curated metadata.
4. Merging new records with existing curated metadata to reach exactly 50 images per species (200 total).
5. Comprehensive post-validation:
   - 200 unique photo IDs.
   - Exactly 50 images per species.
   - All files exist and decode cleanly as RGB JPEGs.
   - Original 36 curated records and files unchanged.
   - All 280 raw images and raw metadata unchanged.
   - Existing caption files completely untouched.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_FinalizeExpansion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_BASELINE_COUNTS = {
    "Tiger": 17,
    "Lion": 10,
    "Asian Elephant": 4,
    "Chital": 5,
}

EXPECTED_EXPANSION_KEEP = {
    "Tiger": 33,
    "Lion": 40,
    "Asian Elephant": 46,
    "Chital": 45,
}

FINAL_TARGET_PER_SPECIES = 50
TOTAL_FINAL_IMAGES = 200

SPECIES_FOLDER_MAP = {
    "Tiger": "tiger",
    "Lion": "lion",
    "Asian Elephant": "elephant",
    "Chital": "chital",
}

CURATED_METADATA_COLUMNS = [
    "image_id",
    "observation_id",
    "photo_id",
    "species",
    "scientific_name",
    "image_path",
    "image_url",
    "photographer",
    "photo_license",
    "attribution",
    "observation_url",
    "observed_on",
    "original_width",
    "original_height",
]


# ---------------------------------------------------------------------------
# Image Verification Helper
# ---------------------------------------------------------------------------
def verify_image(file_path: str) -> Tuple[bool, str]:
    """Verify that file exists, can be opened, is format JPEG, and mode RGB."""
    if not file_path or not os.path.isfile(file_path):
        return False, "File does not exist on disk"
    try:
        with Image.open(file_path) as img:
            if img.format != "JPEG":
                return False, f"Expected JPEG, got {img.format}"
            if img.mode != "RGB":
                return False, f"Expected RGB mode, got {img.mode}"
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main Finalization Routine
# ---------------------------------------------------------------------------
def finalize_expansion(base_dir: str = ".") -> Dict[str, Any]:
    base_dir = os.path.abspath(base_dir)
    dataset_dir = os.path.join(base_dir, "dataset")
    raw_dir = os.path.join(dataset_dir, "raw")
    curated_dir = os.path.join(dataset_dir, "curated")
    curation_dir = os.path.join(dataset_dir, "curation")

    raw_meta_path = os.path.join(dataset_dir, "metadata.csv")
    curated_meta_path = os.path.join(curated_dir, "metadata.csv")
    expansion_status_path = os.path.join(curation_dir, "expansion_curation_status.csv")

    logger.info("============================================================")
    logger.info("   WildMatch AI - Finalizing Expanded Curated Dataset       ")
    logger.info("============================================================")

    # 1. Pre-flight verification
    logger.info("Step 1: Validating pre-conditions and files...")
    if not os.path.isfile(raw_meta_path):
        raise FileNotFoundError(f"Raw metadata not found: {raw_meta_path}")
    if not os.path.isfile(curated_meta_path):
        raise FileNotFoundError(f"Existing curated metadata not found: {curated_meta_path}")
    if not os.path.isfile(expansion_status_path):
        raise FileNotFoundError(f"Expansion curation status not found: {expansion_status_path}")

    raw_meta_df = pd.read_csv(raw_meta_path)
    curated_meta_df = pd.read_csv(curated_meta_path)
    expansion_status_df = pd.read_csv(expansion_status_path)

    # Verify baseline curated dataset
    logger.info("Verifying baseline curated metadata (36 records)...")
    baseline_counts = curated_meta_df["species"].value_counts().to_dict()
    for sp, exp in EXPECTED_BASELINE_COUNTS.items():
        actual = baseline_counts.get(sp, 0)
        if actual != exp:
            raise ValueError(f"Baseline mismatch for {sp}: expected {exp}, got {actual}")
    if len(curated_meta_df) != 36:
        raise ValueError(f"Baseline total mismatch: expected 36, got {len(curated_meta_df)}")

    # Verify baseline image files
    for idx, row in curated_meta_df.iterrows():
        abs_p = os.path.normpath(os.path.join(base_dir, row["image_path"]))
        ok, err = verify_image(abs_p)
        if not ok:
            raise RuntimeError(f"Corrupt or missing baseline curated image '{abs_p}': {err}")
    logger.info("All 36 baseline curated images verified.")

    # Verify expansion status decisions
    logger.info("Verifying expansion curation decisions (164 KEEP records)...")
    expansion_keep_df = expansion_status_df[expansion_status_df["curation_status"] == "KEEP"].copy()
    expansion_counts = expansion_keep_df["species"].value_counts().to_dict()
    for sp, exp in EXPECTED_EXPANSION_KEEP.items():
        actual = expansion_counts.get(sp, 0)
        if actual != exp:
            raise ValueError(f"Expansion KEEP mismatch for {sp}: expected {exp}, got {actual}")
    if len(expansion_keep_df) != 164:
        raise ValueError(f"Expansion KEEP total mismatch: expected 164, got {len(expansion_keep_df)}")

    # Check zero overlap between baseline curated and expansion KEEP photo IDs
    baseline_photo_ids: Set[int] = set(curated_meta_df["photo_id"].astype(int))
    expansion_photo_ids: Set[int] = set(expansion_keep_df["photo_id"].astype(int))
    overlap = baseline_photo_ids.intersection(expansion_photo_ids)
    if overlap:
        raise ValueError(f"Overlap detected between baseline and expansion KEEP photo IDs: {overlap}")
    logger.info("Zero overlap confirmed between baseline and expansion KEEP photo IDs.")

    # 2. Create safety backup of existing curated metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    curated_meta_backup = f"{curated_meta_path}.bak_{timestamp}"
    shutil.copy2(curated_meta_path, curated_meta_backup)
    logger.info(f"Created safety backup of curated metadata at: {curated_meta_backup}")

    # Map raw metadata by photo_id
    raw_meta_by_id = {int(r["photo_id"]): r for _, r in raw_meta_df.iterrows()}

    # 3. Copy (never move) the 164 approved images
    logger.info("Step 2: Copying 164 approved images into dataset/curated/{species}/...")
    new_curated_entries: List[Dict[str, Any]] = []
    copied_files_count = 0

    for idx, row in expansion_keep_df.iterrows():
        photo_id = int(row["photo_id"])
        species = row["species"]

        if photo_id not in raw_meta_by_id:
            raise KeyError(f"Photo ID {photo_id} not found in raw metadata {raw_meta_path}")

        raw_meta_row = raw_meta_by_id[photo_id].to_dict()
        raw_rel_path = raw_meta_row["image_path"]
        raw_abs_path = os.path.normpath(os.path.join(base_dir, raw_rel_path))

        # Verify source raw image
        ok, err = verify_image(raw_abs_path)
        if not ok:
            raise RuntimeError(f"Raw source image invalid '{raw_abs_path}': {err}")

        # Destination path
        folder_name = SPECIES_FOLDER_MAP[species]
        dest_folder = os.path.join(curated_dir, folder_name)
        os.makedirs(dest_folder, exist_ok=True)

        filename = os.path.basename(raw_abs_path)
        dest_abs_path = os.path.join(dest_folder, filename)
        dest_rel_path = f"dataset/curated/{folder_name}/{filename}"

        # Copy image file (preserving raw original)
        shutil.copy2(raw_abs_path, dest_abs_path)
        copied_files_count += 1

        # Verify destination file
        dest_ok, dest_err = verify_image(dest_abs_path)
        if not dest_ok:
            raise RuntimeError(f"Failed verification of copied image '{dest_abs_path}': {dest_err}")

        # Verify raw image still exists at source (confirming copy, not move)
        if not os.path.isfile(raw_abs_path):
            raise RuntimeError(f"Source raw image missing after copy! Critical error: '{raw_abs_path}'")

        # Create curated metadata entry preserving all raw attributes
        curated_entry = dict(raw_meta_row)
        curated_entry["image_path"] = dest_rel_path
        new_curated_entries.append(curated_entry)

    logger.info(f"Successfully copied and verified {copied_files_count} new images.")

    # 4. Merge metadata
    logger.info("Step 3: Merging curated metadata records...")
    existing_entries = curated_meta_df.to_dict("records")
    merged_entries = existing_entries + new_curated_entries

    final_curated_df = pd.DataFrame(merged_entries)[CURATED_METADATA_COLUMNS]

    # Deduplicate by photo_id
    final_curated_df = final_curated_df.drop_duplicates(subset=["photo_id"], keep="first")
    final_curated_df.to_csv(curated_meta_path, index=False)
    logger.info(f"Saved merged curated metadata to: {curated_meta_path} (Total rows: {len(final_curated_df)})")

    # 5. Comprehensive Post-Validation
    logger.info("Step 4: Running comprehensive post-validation...")
    validation_results: Dict[str, Any] = {}

    # Check row count
    validation_results["total_records"] = len(final_curated_df)
    validation_results["total_is_200"] = len(final_curated_df) == TOTAL_FINAL_IMAGES

    # Check unique photo IDs
    validation_results["unique_photo_ids"] = final_curated_df["photo_id"].nunique()
    validation_results["unique_is_200"] = validation_results["unique_photo_ids"] == TOTAL_FINAL_IMAGES

    # Check 50 per species
    final_species_counts = final_curated_df["species"].value_counts().to_dict()
    validation_results["species_counts"] = final_species_counts
    validation_results["species_balanced_50"] = all(
        final_species_counts.get(sp) == FINAL_TARGET_PER_SPECIES for sp in SPECIES_FOLDER_MAP
    )

    # Check all image files on disk
    missing_files = []
    corrupt_files = []
    for idx, row in final_curated_df.iterrows():
        p = os.path.normpath(os.path.join(base_dir, row["image_path"]))
        ok, err = verify_image(p)
        if not ok:
            if "does not exist" in err:
                missing_files.append((row["image_path"], err))
            else:
                corrupt_files.append((row["image_path"], err))

    validation_results["missing_curated_images"] = len(missing_files)
    validation_results["corrupt_curated_images"] = len(corrupt_files)
    validation_results["all_curated_images_valid"] = (
        len(missing_files) == 0 and len(corrupt_files) == 0
    )

    # Check original 36 baseline records unchanged in merged metadata
    baseline_check = curated_meta_df.merge(
        final_curated_df,
        on="photo_id",
        suffixes=("_orig", "_final"),
    )
    validation_results["baseline_records_preserved"] = len(baseline_check) == 36
    diff_fields = 0
    for col in ["species", "photographer", "photo_license", "attribution", "image_path"]:
        diffs = (baseline_check[f"{col}_orig"] != baseline_check[f"{col}_final"]).sum()
        diff_fields += diffs
    validation_results["baseline_fields_identical"] = diff_fields == 0

    # Check raw images and metadata intact
    validation_results["raw_metadata_records"] = len(raw_meta_df)
    validation_results["raw_metadata_intact"] = len(raw_meta_df) == 280
    raw_missing = 0
    for idx, row in raw_meta_df.iterrows():
        if not os.path.isfile(os.path.normpath(os.path.join(base_dir, row["image_path"]))):
            raw_missing += 1
    validation_results["raw_images_missing"] = raw_missing
    validation_results["raw_images_intact"] = raw_missing == 0

    # Check caption files intact
    captions_pilot_path = os.path.join(dataset_dir, "captions", "pilot_captions.csv")
    captions_proc_path = os.path.join(dataset_dir, "captions", "processed_captions.csv")
    pilot_len = len(pd.read_csv(captions_pilot_path)) if os.path.isfile(captions_pilot_path) else -1
    proc_len = len(pd.read_csv(captions_proc_path)) if os.path.isfile(captions_proc_path) else -1
    validation_results["pilot_captions_rows"] = pilot_len
    validation_results["processed_captions_rows"] = proc_len
    validation_results["captions_untouched"] = (pilot_len == 36 and proc_len == 36)

    all_passed = (
        validation_results["total_is_200"]
        and validation_results["unique_is_200"]
        and validation_results["species_balanced_50"]
        and validation_results["all_curated_images_valid"]
        and validation_results["baseline_records_preserved"]
        and validation_results["baseline_fields_identical"]
        and validation_results["raw_metadata_intact"]
        and validation_results["raw_images_intact"]
        and validation_results["captions_untouched"]
    )
    validation_results["all_passed"] = all_passed

    logger.info("============================================================")
    logger.info("Final Validation Summary:")
    logger.info(f"  Total curated records   : {validation_results['total_records']} (Target: 200)")
    logger.info(f"  Unique photo IDs        : {validation_results['unique_photo_ids']}")
    logger.info(f"  Species counts          : {validation_results['species_counts']}")
    logger.info(f"  All images valid        : {validation_results['all_curated_images_valid']}")
    logger.info(f"  Baseline 36 preserved   : {validation_results['baseline_records_preserved']}")
    logger.info(f"  Raw 280 intact          : {validation_results['raw_images_intact']}")
    logger.info(f"  Captions untouched (36) : {validation_results['captions_untouched']}")
    logger.info(f"  Overall Validation      : {validation_results['all_passed']}")
    logger.info("============================================================")

    if not all_passed:
        logger.error("Finalization failed post-validation checks! Halting.")
        sys.exit(1)

    return validation_results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Finalize expanded curated dataset for WildMatch AI."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory).",
    )
    args = parser.parse_args()
    finalize_expansion(base_dir=args.base_dir)


if __name__ == "__main__":
    main()
