"""
WildMatch AI - Manual Visual Curation Workflow

This module provides:
1. Contact sheet generation for manual inspection of raw images by species.
2. Initializing and updating `dataset/curation/curation_status.csv` with safe reruns.
3. Finalization routine to copy approved `KEEP` images into `dataset/curated/`
   and generate `dataset/curated/metadata.csv` without modifying raw data.
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
logger = logging.getLogger("WildMatch_Curator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_STATUSES = {"KEEP", "REJECT", "REVIEW"}

CURATION_COLUMNS = [
    "image_id",
    "photo_id",
    "species",
    "image_path",
    "curation_status",
    "rejection_reason",
    "reviewer_notes",
]

SPECIES_FOLDER_MAP = {
    "Tiger": "tiger",
    "Lion": "lion",
    "Asian Elephant": "elephant",
    "Chital": "chital",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def resolve_path(base_dir: str, rel_path: str) -> str:
    """Resolve a relative image path against base_dir consistently."""
    if not rel_path or not isinstance(rel_path, str):
        return ""
    if os.path.isabs(rel_path):
        return rel_path
    direct = os.path.normpath(os.path.join(base_dir, rel_path))
    if os.path.isfile(direct):
        return direct
    fallback = os.path.normpath(os.path.join(base_dir, "dataset", rel_path))
    if os.path.isfile(fallback):
        return fallback
    return direct


def verify_image(file_path: str) -> Tuple[bool, str]:
    """Verify that file exists and can be fully opened/decoded by Pillow."""
    if not file_path or not os.path.isfile(file_path):
        return False, "File not found on disk"
    try:
        with Image.open(file_path) as img:
            if img.format != "JPEG":
                return False, f"Format is {img.format}, expected JPEG"
            if img.mode != "RGB":
                return False, f"Mode is {img.mode}, expected RGB"
            img.load()
        return True, ""
    except Exception as e:
        return False, f"Pillow decode error: {str(e)}"


def get_font(size: int = 14) -> ImageFont.ImageFont:
    """Safely obtain a TrueType font if available on Windows, else fallback to default."""
    font_candidates = [
        "arial.ttf",
        "segoeui.ttf",
        "calibri.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Contact Sheet Generator
# ---------------------------------------------------------------------------

def generate_contact_sheet(
    species_name: str,
    records: List[Dict[str, Any]],
    base_dir: str,
    output_path: str,
    cols: int = 5,
    cell_w: int = 380,
    cell_h: int = 340,
    thumb_w: int = 360,
    thumb_h: int = 260,
) -> None:
    """
    Generate a high-resolution contact sheet grid for a given species.
    Displays filename, photo ID, and original dimensions below each thumbnail.
    """
    if not records:
        logger.warning(f"No records provided for contact sheet of species '{species_name}'.")
        return

    n_images = len(records)
    cols = min(cols, n_images) if n_images > 0 else 1
    rows = math.ceil(n_images / cols)

    header_h = 90
    margin = 25
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    total_w = grid_w + 2 * margin
    total_h = grid_h + header_h + margin

    # Create sheet with clean dark-slate background
    sheet = Image.new("RGB", (total_w, total_h), color=(24, 28, 35))
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(26)
    subtitle_font = get_font(15)
    label_font = get_font(13)
    dim_font = get_font(11)

    # Draw header banner
    draw.text(
        (margin, 20),
        f"WildMatch AI Curation Sheet: {species_name}",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (margin, 56),
        f"Total images: {n_images} | Inspect visual quality, framing, and occlusions before marking KEEP/REJECT in curation_status.csv",
        fill=(170, 185, 200),
        font=subtitle_font,
    )
    # Header separator
    draw.line([(margin, 82), (total_w - margin, 82)], fill=(50, 60, 75), width=2)

    for idx, rec in enumerate(records):
        col = idx % cols
        row = idx // cols

        x = margin + col * cell_w
        y = header_h + row * cell_h

        card_rect = [x + 5, y + 5, x + cell_w - 5, y + cell_h - 5]
        draw.rectangle(card_rect, fill=(35, 41, 52), outline=(55, 65, 80), width=1)

        img_path = resolve_path(base_dir, rec.get("image_path", ""))
        photo_id = rec.get("photo_id", "N/A")
        orig_w = rec.get("original_width", "?")
        orig_h = rec.get("original_height", "?")
        filename = os.path.basename(img_path)

        # Thumbnail rendering preserving aspect ratio
        if os.path.isfile(img_path):
            try:
                with Image.open(img_path) as orig_img:
                    thumb = orig_img.copy()
                    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    tw, th = thumb.size

                    # Center thumbnail in thumbnail box
                    tx = x + 10 + (thumb_w - tw) // 2
                    ty = y + 10 + (thumb_h - th) // 2
                    sheet.paste(thumb, (tx, ty))
            except Exception as e:
                err_text = f"Corrupt image: {e}"
                draw.text((x + 15, y + thumb_h // 2), err_text, fill=(255, 100, 100), font=label_font)
        else:
            draw.text((x + 15, y + thumb_h // 2), "File Missing", fill=(255, 80, 80), font=label_font)

        # Text labels below image
        label_y = y + thumb_h + 16
        draw.text((x + 12, label_y), filename, fill=(240, 245, 255), font=label_font)
        draw.text(
            (x + 12, label_y + 18),
            f"Photo ID: {photo_id} | {orig_w}x{orig_h}",
            fill=(140, 155, 175),
            font=dim_font,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=95)
    logger.info(f"Saved contact sheet for {species_name} ({n_images} images) to: {output_path}")


# ---------------------------------------------------------------------------
# Curation Status Management
# ---------------------------------------------------------------------------

def load_or_init_curation_status(
    curation_csv_path: str,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load existing curation_status.csv if present, preserving existing manual
    statuses and reviewer notes. Initialize missing records with status 'REVIEW'.
    """
    existing_status_map: Dict[int, Dict[str, str]] = {}

    if os.path.exists(curation_csv_path):
        try:
            curr_df = pd.read_csv(curation_csv_path)
            for _, row in curr_df.iterrows():
                p_id = row.get("photo_id")
                if pd.notna(p_id) and str(p_id).isdigit():
                    int_id = int(p_id)
                    existing_status_map[int_id] = {
                        "curation_status": str(row.get("curation_status", "REVIEW")).strip().upper(),
                        "rejection_reason": str(row.get("rejection_reason", "") if pd.notna(row.get("rejection_reason")) else ""),
                        "reviewer_notes": str(row.get("reviewer_notes", "") if pd.notna(row.get("reviewer_notes")) else ""),
                    }
            logger.info(f"Loaded existing curation decisions for {len(existing_status_map)} images.")
        except Exception as e:
            logger.warning(f"Could not parse existing curation_status.csv ({e}). Preserving backup.")
            backup_path = f"{curation_csv_path}.bak"
            shutil.copy2(curation_csv_path, backup_path)

    reconciled_rows: List[Dict[str, Any]] = []
    for _, row in metadata_df.iterrows():
        p_id = int(row["photo_id"])
        image_id = str(row["image_id"])
        species = str(row["species"])
        image_path = str(row["image_path"])

        if p_id in existing_status_map:
            prev = existing_status_map[p_id]
            status = prev["curation_status"] if prev["curation_status"] in ALLOWED_STATUSES else "REVIEW"
            rejection_reason = prev["rejection_reason"]
            reviewer_notes = prev["reviewer_notes"]
        else:
            status = "REVIEW"
            rejection_reason = ""
            reviewer_notes = ""

        reconciled_rows.append({
            "image_id": image_id,
            "photo_id": p_id,
            "species": species,
            "image_path": image_path,
            "curation_status": status,
            "rejection_reason": rejection_reason,
            "reviewer_notes": reviewer_notes,
        })

    curation_df = pd.DataFrame(reconciled_rows)[CURATION_COLUMNS]
    curation_df = curation_df.drop_duplicates(subset=["photo_id"], keep="first")
    return curation_df


# ---------------------------------------------------------------------------
# Finalization Operation
# ---------------------------------------------------------------------------

def finalize_curation(base_dir: str) -> None:
    """
    Finalize the curated dataset:
    1. Verify that all records in curation_status.csv have been decided (KEEP or REJECT).
    2. Refuses to finalize if any record has status 'REVIEW'.
    3. Copies KEEP images into dataset/curated/{species}/ without touching raw images.
    4. Creates dataset/curated/metadata.csv with all original metadata fields.
    """
    metadata_csv_path = os.path.join(base_dir, "dataset", "metadata.csv")
    curation_csv_path = os.path.join(base_dir, "dataset", "curation", "curation_status.csv")
    curated_dir = os.path.join(base_dir, "dataset", "curated")
    curated_meta_path = os.path.join(curated_dir, "metadata.csv")

    if not os.path.exists(curation_csv_path):
        raise FileNotFoundError(f"Curation status file not found: {curation_csv_path}")
    if not os.path.exists(metadata_csv_path):
        raise FileNotFoundError(f"Source metadata file not found: {metadata_csv_path}")

    curation_df = pd.read_csv(curation_csv_path)
    meta_df = pd.read_csv(metadata_csv_path)

    # 1. Check for invalid or undecided statuses
    pending_reviews = curation_df[curation_df["curation_status"].str.upper() == "REVIEW"]
    if not pending_reviews.empty:
        logger.error("=" * 60)
        logger.error(f"FINALIZATION REFUSED: {len(pending_reviews)} images still have status 'REVIEW'.")
        logger.error("All images must be explicitly reviewed and set to 'KEEP' or 'REJECT' before finalization.")
        logger.error("-" * 60)
        for _, r in pending_reviews.head(10).iterrows():
            logger.error(f"  - [{r['species']}] Photo ID {r['photo_id']} ({r['image_path']})")
        if len(pending_reviews) > 10:
            logger.error(f"  ... and {len(pending_reviews) - 10} more.")
        logger.error("=" * 60)
        sys.exit(1)

    invalid_statuses = curation_df[~curation_df["curation_status"].str.upper().isin(ALLOWED_STATUSES)]
    if not invalid_statuses.empty:
        logger.error(f"FINALIZATION REFUSED: Found invalid statuses: {invalid_statuses['curation_status'].unique()}")
        sys.exit(1)

    keep_records = curation_df[curation_df["curation_status"].str.upper() == "KEEP"]
    reject_records = curation_df[curation_df["curation_status"].str.upper() == "REJECT"]

    logger.info("=" * 60)
    logger.info("        FINALIZING CURATED DATASET        ")
    logger.info("=" * 60)
    logger.info(f"Total reviewed images: {len(curation_df)}")
    logger.info(f"Approved (KEEP):       {len(keep_records)}")
    logger.info(f"Rejected (REJECT):     {len(reject_records)}")
    logger.info("=" * 60)

    if keep_records.empty:
        logger.warning("No images marked as KEEP. Finalization aborted.")
        return

    # Check duplicate photo IDs
    duplicate_ids = keep_records[keep_records.duplicated(subset=["photo_id"], keep=False)]
    if not duplicate_ids.empty:
        logger.error(f"Duplicate photo IDs detected in KEEP records: {duplicate_ids['photo_id'].tolist()}")
        sys.exit(1)

    os.makedirs(curated_dir, exist_ok=True)
    meta_by_id = {int(r["photo_id"]): r for _, r in meta_df.iterrows()}

    curated_meta_rows: List[Dict[str, Any]] = []

    for _, k_row in keep_records.iterrows():
        p_id = int(k_row["photo_id"])
        species_name = str(k_row["species"])
        raw_rel_path = str(k_row["image_path"])
        raw_abs_path = resolve_path(base_dir, raw_rel_path)

        if p_id not in meta_by_id:
            logger.error(f"Missing original metadata for photo_id {p_id}. Skipping.")
            continue

        # Verify source file before copying
        is_ok, err_msg = verify_image(raw_abs_path)
        if not is_ok:
            logger.error(f"Cannot copy corrupt or missing source file '{raw_abs_path}': {err_msg}")
            sys.exit(1)

        # Destination path: dataset/curated/<folder>/<filename>
        folder_name = SPECIES_FOLDER_MAP.get(species_name, species_name.lower().replace(" ", "_"))
        target_species_dir = os.path.join(curated_dir, folder_name)
        os.makedirs(target_species_dir, exist_ok=True)

        filename = os.path.basename(raw_abs_path)
        dest_abs_path = os.path.join(target_species_dir, filename)
        dest_rel_path = os.path.relpath(dest_abs_path, start=base_dir).replace("\\", "/")

        # Copy safely without modifying raw images
        shutil.copy2(raw_abs_path, dest_abs_path)

        # Verify copied destination file
        dest_ok, dest_err = verify_image(dest_abs_path)
        if not dest_ok:
            logger.error(f"Destination image verification failed for '{dest_abs_path}': {dest_err}")
            sys.exit(1)

        # Construct curated metadata row from original metadata row
        orig_meta = meta_by_id[p_id].to_dict()
        curated_entry = dict(orig_meta)
        curated_entry["image_path"] = dest_rel_path
        curated_meta_rows.append(curated_entry)

    curated_meta_df = pd.DataFrame(curated_meta_rows)[meta_df.columns]
    curated_meta_df = curated_meta_df.drop_duplicates(subset=["photo_id"], keep="first")
    curated_meta_df.to_csv(curated_meta_path, index=False)

    logger.info("-" * 60)
    logger.info(f"Curated metadata saved to: {curated_meta_path} (Total: {len(curated_meta_df)} rows)")
    logger.info("Curated counts by species:")
    for sp, cnt in curated_meta_df["species"].value_counts().items():
        logger.info(f"  - {sp:18s}: {cnt} images")
    logger.info("=" * 60)
    logger.info("Finalization completed successfully!")


# ---------------------------------------------------------------------------
# Workflow Orchestration
# ---------------------------------------------------------------------------

def run_workflow(
    base_dir: str = ".",
    species_filter: Optional[List[str]] = None,
    finalize_mode: bool = False,
) -> None:
    """Run contact sheet generation or finalization based on CLI flags."""
    base_dir = os.path.abspath(base_dir)

    if finalize_mode:
        finalize_curation(base_dir)
        return

    # Generation & Initialization Mode
    metadata_csv_path = os.path.join(base_dir, "dataset", "metadata.csv")
    curation_dir = os.path.join(base_dir, "dataset", "curation")
    contact_sheet_dir = os.path.join(curation_dir, "contact_sheets")
    curation_csv_path = os.path.join(curation_dir, "curation_status.csv")

    if not os.path.exists(metadata_csv_path):
        logger.error(f"Metadata file not found: {metadata_csv_path}")
        sys.exit(1)

    meta_df = pd.read_csv(metadata_csv_path)
    logger.info(f"Loaded {len(meta_df)} verified records from: {metadata_csv_path}")

    # Validate all referenced images
    logger.info("Validating referenced image files on disk...")
    invalid_files = []
    for _, row in meta_df.iterrows():
        p = resolve_path(base_dir, str(row["image_path"]))
        ok, err = verify_image(p)
        if not ok:
            invalid_files.append((row["photo_id"], p, err))

    if invalid_files:
        logger.error(f"Found {len(invalid_files)} invalid/missing image files in metadata!")
        for pid, p, err in invalid_files:
            logger.error(f"  Photo {pid}: {p} -> {err}")
        sys.exit(1)
    logger.info("All 55 referenced images verified successfully on disk.")

    # Generate or update curation_status.csv
    curation_df = load_or_init_curation_status(curation_csv_path, meta_df)
    curation_df.to_csv(curation_csv_path, index=False)
    logger.info(f"Curation status CSV updated: {curation_csv_path} (Total: {len(curation_df)} records)")

    # Status counts breakdown
    logger.info("Current curation status breakdown:")
    for status, count in curation_df["curation_status"].value_counts().items():
        logger.info(f"  - {status:8s}: {count}")

    # Generate contact sheets per species
    selected_species = meta_df["species"].unique()
    if species_filter:
        lower_filter = [s.strip().lower() for s in species_filter]
        selected_species = [s for s in selected_species if s.lower() in lower_filter]

    os.makedirs(contact_sheet_dir, exist_ok=True)
    for sp in selected_species:
        sp_records = meta_df[meta_df["species"] == sp].to_dict("records")
        sp_folder = SPECIES_FOLDER_MAP.get(sp, sp.lower().replace(" ", "_"))
        out_sheet = os.path.join(contact_sheet_dir, f"{sp_folder}_contact_sheet.jpg")
        generate_contact_sheet(
            species_name=sp,
            records=sp_records,
            base_dir=base_dir,
            output_path=out_sheet,
            cols=5,
        )

    logger.info("=" * 60)
    logger.info("Curation preparation complete!")
    logger.info(f"1. Contact sheets: {contact_sheet_dir}")
    logger.info(f"2. Curation CSV:   {curation_csv_path}")
    logger.info("Next step: Review images, set KEEP/REJECT in curation_status.csv, then run with --finalize.")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WildMatch AI - Manual Visual Curation Workflow"
    )
    parser.add_argument(
        "--generate-sheets",
        action="store_true",
        default=True,
        help="Generate contact sheets and initialize curation_status.csv (default)",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        default=False,
        help="Finalize curated dataset by copying KEEP images and generating curated/metadata.csv",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Optional species filter for contact sheets (e.g. tiger lion)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    run_workflow(
        base_dir=args.base_dir,
        species_filter=args.species,
        finalize_mode=args.finalize,
    )


if __name__ == "__main__":
    main()
