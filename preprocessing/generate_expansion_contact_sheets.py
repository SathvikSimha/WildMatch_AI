"""
WildMatch AI - Expansion Contact Sheet Generator & Status Initialization

This module handles:
1. Identifying the 225 newly collected raw images by excluding the original 55 photo IDs.
2. Initializing `dataset/curation/expansion_curation_status.csv` with all 225 records set to 'REVIEW'.
3. Generating high-resolution contact sheets under `dataset/curation/expansion_contact_sheets/`
   for visual curation (Tiger, Lion, Asian Elephant, Chital).
4. Aspect-ratio preservation, large thumbnails, clear metadata labels, and multi-sheet pagination.
5. Strict validation ensuring no duplicates, no missing images, and zero modification of original assets.
"""

import argparse
import logging
import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_ExpansionContactSheets")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NEW_TOTAL = 225
EXPECTED_NEW_PER_SPECIES = {
    "Tiger": 45,
    "Lion": 60,
    "Asian Elephant": 60,
    "Chital": 60,
}

SPECIES_PREFIX_MAP = {
    "Tiger": "tiger",
    "Lion": "lion",
    "Asian Elephant": "elephant",
    "Chital": "chital",
}

SPECIES_SCIENTIFIC_MAP = {
    "Tiger": "Panthera tigris",
    "Lion": "Panthera leo",
    "Asian Elephant": "Elephas maximus",
    "Chital": "Axis axis",
}

EXPANSION_STATUS_COLUMNS = [
    "image_id",
    "photo_id",
    "species",
    "image_path",
    "curation_status",
    "rejection_reason",
    "reviewer_notes",
]

IMAGES_PER_SHEET = 20  # 5 columns x 4 rows per sheet


# ---------------------------------------------------------------------------
# Font & Helper Functions
# ---------------------------------------------------------------------------
def get_font(size: int) -> ImageFont.ImageFont:
    """Safely obtain a TrueType font or fallback to default."""
    for candidate in ["arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def verify_image(file_path: str) -> Tuple[bool, str]:
    """Verify that file exists and can be fully decoded by Pillow."""
    if not file_path or not os.path.isfile(file_path):
        return False, "File not found on disk"
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
# Data Identification & Status CSV
# ---------------------------------------------------------------------------
def identify_new_records(
    raw_metadata_path: str,
    baseline_backup_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify exactly the newly downloaded records by subtracting the original 55 photo IDs.
    Returns (new_records_df, original_records_df).
    """
    if not os.path.isfile(raw_metadata_path):
        raise FileNotFoundError(f"Raw metadata not found at: {raw_metadata_path}")
    if not os.path.isfile(baseline_backup_path):
        raise FileNotFoundError(f"Baseline backup metadata not found at: {baseline_backup_path}")

    current_df = pd.read_csv(raw_metadata_path)
    baseline_df = pd.read_csv(baseline_backup_path)

    baseline_photo_ids: Set[int] = set(baseline_df["photo_id"].dropna().astype(int))
    logger.info(f"Loaded baseline dataset containing {len(baseline_photo_ids)} photo IDs.")

    # Filter out baseline photo IDs
    is_new = ~current_df["photo_id"].astype(int).isin(baseline_photo_ids)
    new_df = current_df[is_new].copy()

    logger.info(f"Identified {len(new_df)} new records out of {len(current_df)} total records.")
    return new_df, baseline_df


def create_expansion_curation_status_file(
    new_records_df: pd.DataFrame,
    output_status_path: str,
) -> pd.DataFrame:
    """
    Create dataset/curation/expansion_curation_status.csv with all records initialized as REVIEW.
    """
    status_records = []
    for _, row in new_records_df.iterrows():
        status_records.append({
            "image_id": row["image_id"],
            "photo_id": int(row["photo_id"]),
            "species": row["species"],
            "image_path": row["image_path"],
            "curation_status": "REVIEW",
            "rejection_reason": "",
            "reviewer_notes": "",
        })

    status_df = pd.DataFrame(status_records)[EXPANSION_STATUS_COLUMNS]
    os.makedirs(os.path.dirname(os.path.abspath(output_status_path)), exist_ok=True)
    status_df.to_csv(output_status_path, index=False)
    logger.info(f"Created expansion curation status file at: {output_status_path} (Total rows: {len(status_df)})")
    return status_df


# ---------------------------------------------------------------------------
# Contact Sheet Rendering
# ---------------------------------------------------------------------------
def render_contact_sheet(
    species_name: str,
    sheet_index: int,
    total_sheets_for_species: int,
    records: List[Dict[str, Any]],
    output_path: str,
    base_dir: str = ".",
    cols: int = 5,
    cell_w: int = 380,
    cell_h: int = 360,
    thumb_w: int = 360,
    thumb_h: int = 260,
) -> None:
    """
    Render a single paginated contact sheet for a subset of records.
    """
    n_images = len(records)
    if n_images == 0:
        return

    actual_cols = min(cols, n_images)
    rows = math.ceil(n_images / actual_cols)

    header_h = 90
    margin = 25
    grid_w = actual_cols * cell_w
    grid_h = rows * cell_h
    total_w = grid_w + 2 * margin
    total_h = grid_h + header_h + margin

    sheet = Image.new("RGB", (total_w, total_h), color=(22, 26, 33))
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(24)
    subtitle_font = get_font(14)
    label_font = get_font(13)
    dim_font = get_font(11)

    scientific_name = SPECIES_SCIENTIFIC_MAP.get(species_name, "")

    # Header
    draw.text(
        (margin, 18),
        f"WildMatch AI — Visual Curation: {species_name} ({scientific_name})",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (margin, 52),
        f"Sheet {sheet_index:02d} of {total_sheets_for_species:02d} ({n_images} candidates on this sheet) | "
        f"Mark decisions in dataset/curation/expansion_curation_status.csv [KEEP / REJECT]",
        fill=(160, 180, 205),
        font=subtitle_font,
    )
    draw.line([(margin, 80), (total_w - margin, 80)], fill=(45, 55, 70), width=2)

    for idx, rec in enumerate(records):
        col = idx % actual_cols
        row = idx // actual_cols

        x = margin + col * cell_w
        y = header_h + row * cell_h

        # Card container
        card_rect = [x + 5, y + 5, x + cell_w - 5, y + cell_h - 5]
        draw.rectangle(card_rect, fill=(32, 38, 48), outline=(50, 60, 75), width=1)

        raw_rel_path = rec.get("image_path", "")
        img_abs_path = os.path.normpath(os.path.join(base_dir, raw_rel_path)) if not os.path.isabs(raw_rel_path) else raw_rel_path
        photo_id = rec.get("photo_id", "N/A")
        orig_w = rec.get("original_width", "?")
        orig_h = rec.get("original_height", "?")
        license_code = rec.get("photo_license", "unknown")
        filename = os.path.basename(img_abs_path)

        # Thumbnail rendering preserving aspect ratio
        if os.path.isfile(img_abs_path):
            try:
                with Image.open(img_abs_path) as orig_img:
                    thumb = orig_img.copy()
                    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    tw, th = thumb.size

                    # Center thumbnail in thumbnail region
                    tx = x + 10 + (thumb_w - tw) // 2
                    ty = y + 10 + (thumb_h - th) // 2
                    sheet.paste(thumb, (tx, ty))
            except Exception as e:
                draw.text((x + 15, y + thumb_h // 2), f"Error: {e}", fill=(255, 90, 90), font=label_font)
        else:
            draw.text((x + 15, y + thumb_h // 2), "File Missing", fill=(255, 90, 90), font=label_font)

        # Labels below image
        label_y = y + thumb_h + 16
        draw.text((x + 12, label_y), filename, fill=(245, 248, 255), font=label_font)
        draw.text(
            (x + 12, label_y + 18),
            f"Photo ID: {photo_id}  |  {orig_w}x{orig_h}",
            fill=(150, 170, 195),
            font=dim_font,
        )
        draw.text(
            (x + 12, label_y + 36),
            f"{species_name} ({scientific_name})  |  {license_code}",
            fill=(115, 195, 255),
            font=dim_font,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=95)
    logger.info(f"Saved contact sheet: {output_path} ({n_images} images)")


def generate_all_expansion_contact_sheets(
    new_records_df: pd.DataFrame,
    output_dir: str,
    base_dir: str = ".",
    images_per_sheet: int = IMAGES_PER_SHEET,
) -> List[Dict[str, Any]]:
    """
    Generate all contact sheets for all species, split across multiple sheets where needed.
    """
    sheet_metadata: List[Dict[str, Any]] = []

    for species, expected_count in EXPECTED_NEW_PER_SPECIES.items():
        species_df = new_records_df[new_records_df["species"] == species].copy()
        actual_count = len(species_df)

        if actual_count != expected_count:
            logger.warning(
                f"Species {species}: Expected {expected_count} records, found {actual_count}."
            )

        # Sort consistently by photo_id
        species_df = species_df.sort_values(by="photo_id").reset_index(drop=True)
        records = species_df.to_dict("records")

        total_sheets = math.ceil(len(records) / images_per_sheet)
        prefix = SPECIES_PREFIX_MAP[species]

        logger.info(
            f"Generating contact sheets for {species}: {len(records)} images across {total_sheets} sheet(s)..."
        )

        for sheet_idx in range(1, total_sheets + 1):
            start_idx = (sheet_idx - 1) * images_per_sheet
            end_idx = min(start_idx + images_per_sheet, len(records))
            sheet_records = records[start_idx:end_idx]

            sheet_filename = f"{prefix}_{sheet_idx:02d}.jpg"
            sheet_path = os.path.join(output_dir, sheet_filename)

            render_contact_sheet(
                species_name=species,
                sheet_index=sheet_idx,
                total_sheets_for_species=total_sheets,
                records=sheet_records,
                output_path=sheet_path,
                base_dir=base_dir,
            )

            sheet_metadata.append({
                "species": species,
                "sheet_filename": sheet_filename,
                "sheet_path": sheet_path.replace("\\", "/"),
                "sheet_number": sheet_idx,
                "total_sheets": total_sheets,
                "image_count": len(sheet_records),
                "photo_ids": [r["photo_id"] for r in sheet_records],
            })

    return sheet_metadata


# ---------------------------------------------------------------------------
# Comprehensive Validation Routine
# ---------------------------------------------------------------------------
def validate_expansion_setup(
    new_records_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    status_df: pd.DataFrame,
    sheet_metadata: List[Dict[str, Any]],
    base_dir: str = ".",
) -> Dict[str, Any]:
    """
    Perform comprehensive validation of expansion contact sheets and status tracking.
    """
    results: Dict[str, Any] = {}

    # Check 1: Record counts
    total_new = len(new_records_df)
    results["total_new_records"] = total_new
    results["exact_225_new"] = total_new == EXPECTED_NEW_TOTAL

    # Check 2: Species distribution
    species_counts = new_records_df["species"].value_counts().to_dict()
    results["species_distribution"] = species_counts
    dist_correct = all(species_counts.get(sp, 0) == count for sp, count in EXPECTED_NEW_PER_SPECIES.items())
    results["species_distribution_correct"] = dist_correct

    # Check 3: File decoding & existence for all new records
    file_errors = []
    for _, row in new_records_df.iterrows():
        rel_p = row["image_path"]
        abs_p = os.path.normpath(os.path.join(base_dir, rel_p)) if not os.path.isabs(rel_p) else rel_p
        ok, err = verify_image(abs_p)
        if not ok:
            file_errors.append((rel_p, err))
    results["file_errors"] = file_errors
    results["all_files_valid"] = len(file_errors) == 0

    # Check 4: No overlap with baseline 55 records
    baseline_ids = set(baseline_df["photo_id"].dropna().astype(int))
    new_ids = set(new_records_df["photo_id"].dropna().astype(int))
    overlap = baseline_ids.intersection(new_ids)
    results["baseline_overlap_count"] = len(overlap)
    results["no_baseline_overlap"] = len(overlap) == 0

    # Check 5: Every new image appears exactly once across contact sheets
    all_sheet_photo_ids = []
    for sm in sheet_metadata:
        all_sheet_photo_ids.extend(sm["photo_ids"])

    results["total_images_in_sheets"] = len(all_sheet_photo_ids)
    results["unique_images_in_sheets"] = len(set(all_sheet_photo_ids))
    results["sheet_coverage_complete"] = (
        len(all_sheet_photo_ids) == EXPECTED_NEW_TOTAL
        and set(all_sheet_photo_ids) == new_ids
    )

    # Check 6: Status file initialization
    status_total = len(status_df)
    status_counts = status_df["curation_status"].value_counts().to_dict()
    results["status_total"] = status_total
    results["status_all_review"] = status_counts == {"REVIEW": EXPECTED_NEW_TOTAL}

    results["all_passed"] = (
        results["exact_225_new"]
        and results["species_distribution_correct"]
        and results["all_files_valid"]
        and results["no_baseline_overlap"]
        and results["sheet_coverage_complete"]
        and results["status_all_review"]
    )

    return results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate contact sheets and curation status for 225 newly collected images."
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset/metadata.csv",
        help="Path to current full raw metadata CSV (default: dataset/metadata.csv).",
    )
    parser.add_argument(
        "--baseline-backup-csv",
        type=str,
        default="dataset/metadata.csv.bak_expansion_20261105_210717",
        help="Path to pre-expansion baseline metadata backup CSV.",
    )
    parser.add_argument(
        "--output-contact-sheets-dir",
        type=str,
        default="dataset/curation/expansion_contact_sheets",
        help="Directory to save generated contact sheets.",
    )
    parser.add_argument(
        "--output-status-csv",
        type=str,
        default="dataset/curation/expansion_curation_status.csv",
        help="Path to save expansion curation status CSV.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory).",
    )

    args = parser.parse_args()

    logger.info("============================================================")
    logger.info("   WildMatch AI - Expansion Contact Sheet Generation        ")
    logger.info("============================================================")

    # 1. Identify 225 new records
    new_records_df, baseline_df = identify_new_records(
        raw_metadata_path=args.metadata_csv,
        baseline_backup_path=args.baseline_backup_csv,
    )

    # 2. Create expansion curation status CSV
    status_df = create_expansion_curation_status_file(
        new_records_df=new_records_df,
        output_status_path=args.output_status_csv,
    )

    # 3. Generate contact sheets
    sheet_metadata = generate_all_expansion_contact_sheets(
        new_records_df=new_records_df,
        output_dir=args.output_contact_sheets_dir,
        base_dir=args.base_dir,
    )

    # 4. Perform validation
    validation = validate_expansion_setup(
        new_records_df=new_records_df,
        baseline_df=baseline_df,
        status_df=status_df,
        sheet_metadata=sheet_metadata,
        base_dir=args.base_dir,
    )

    logger.info("============================================================")
    logger.info("Validation Results:")
    logger.info(f"  Exact 225 new records       : {validation['exact_225_new']}")
    logger.info(f"  Species distribution correct: {validation['species_distribution_correct']} ({validation['species_distribution']})")
    logger.info(f"  All image files valid       : {validation['all_files_valid']} (Errors: {len(validation['file_errors'])})")
    logger.info(f"  No baseline overlap         : {validation['no_baseline_overlap']}")
    logger.info(f"  Sheet coverage complete     : {validation['sheet_coverage_complete']} ({validation['total_images_in_sheets']} rendered)")
    logger.info(f"  Status initialized to REVIEW: {validation['status_all_review']}")
    logger.info(f"  Overall Validation Passed   : {validation['all_passed']}")
    logger.info("============================================================")

    if not validation["all_passed"]:
        logger.error("Validation failed! Halting.")
        sys.exit(1)

    print("\nGenerated Expansion Contact Sheets:")
    for sm in sheet_metadata:
        print(f"  - [{sm['species']}] {sm['sheet_filename']}: {sm['image_count']} images -> {sm['sheet_path']}")


if __name__ == "__main__":
    main()
