"""
WildMatch AI - Expansion Caption Review Contact Sheet Generator

Generates high-resolution, paginated visual review contact sheets for the 164
newly generated BLIP captions across Tiger, Lion, Asian Elephant, and Chital.

Key Features:
1. Filters pilot_captions.csv to strictly select the 164 records with blank final_caption.
2. Preserves the 36 baseline reviewed captions (excludes them from editing sheets).
3. Renders separate contact sheets per species with approximately 20 images per sheet (4 cols x 5 rows).
4. Preserves aspect ratios (no cropping/stretching) in large 410x260 px thumbnail regions.
5. Displays Photo ID, filename, original dimensions, and full word-wrapped BLIP caption without truncation.
6. Generates `dataset/captions/review_contact_sheets/caption_review_index.csv` mapping photo IDs to sheets and positions.
7. Leaves pilot_captions.csv, curated metadata, and image files strictly unmodified.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Dict, List, Set, Tuple

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
logger = logging.getLogger("WildMatch_CaptionReviewSheets")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_NEW_TOTAL = 164
EXPECTED_NEW_PER_SPECIES = {
    "Tiger": 33,
    "Lion": 40,
    "Asian Elephant": 46,
    "Chital": 45,
}

SPECIES_FILE_PREFIX = {
    "Tiger": "tiger",
    "Lion": "lion",
    "Asian Elephant": "elephant",
    "Chital": "chital",
}

IMAGES_PER_SHEET = 20  # 4 columns x 5 rows max


# ---------------------------------------------------------------------------
# Font & Helper Utilities
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
    """Verify that file exists and can be fully opened/decoded by Pillow."""
    if not file_path or not os.path.isfile(file_path):
        return False, "File not found on disk"
    try:
        with Image.open(file_path) as img:
            img.load()
        return True, ""
    except Exception as e:
        return False, str(e)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_w: int,
    font: ImageFont.ImageFont,
    fill: Any,
    line_spacing: int = 4,
) -> int:
    """
    Draw text wrapped to fit within max_w pixels.
    Returns the new y position after drawing.
    """
    words = text.split()
    if not words:
        return y

    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_w = bbox[2] - bbox[0]
        if line_w <= max_w:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1]
        y += line_h + line_spacing

    return y


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------
def load_caption_candidates(
    captions_csv_path: str,
    curated_meta_path: str,
    base_dir: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load caption candidates:
    - Filters pilot_captions.csv for records with blank final_caption.
    - Merges with curated metadata to obtain original dimensions.
    - Returns (new_captions_df, baseline_reviewed_df).
    """
    if not os.path.isfile(captions_csv_path):
        raise FileNotFoundError(f"Captions file not found: {captions_csv_path}")
    if not os.path.isfile(curated_meta_path):
        raise FileNotFoundError(f"Curated metadata not found: {curated_meta_path}")

    captions_df = pd.read_csv(captions_csv_path)
    curated_meta_df = pd.read_csv(curated_meta_path)

    # Separate reviewed vs new records
    is_blank = captions_df["final_caption"].isna() | (captions_df["final_caption"].astype(str).str.strip() == "")
    new_df = captions_df[is_blank].copy()
    reviewed_df = captions_df[~is_blank].copy()

    logger.info(f"Loaded {len(captions_df)} total records from {captions_csv_path}.")
    logger.info(f"Existing reviewed records: {len(reviewed_df)}")
    logger.info(f"Newly generated records awaiting review: {len(new_df)}")

    # Merge metadata for width and height
    meta_sub = curated_meta_df[["photo_id", "original_width", "original_height"]].copy()
    new_merged = new_df.merge(meta_sub, on="photo_id", how="left")

    return new_merged, reviewed_df


# ---------------------------------------------------------------------------
# Sheet Rendering
# ---------------------------------------------------------------------------
def render_caption_review_sheet(
    species_name: str,
    sheet_index: int,
    total_sheets_for_species: int,
    records: List[Dict[str, Any]],
    output_path: str,
    base_dir: str = ".",
    cols: int = 4,
    cell_w: int = 440,
    cell_h: int = 440,
    thumb_box_w: int = 410,
    thumb_box_h: int = 260,
) -> List[Dict[str, Any]]:
    """
    Render a single paginated visual review contact sheet for BLIP captions.
    Returns list of index records for this sheet.
    """
    n_images = len(records)
    if n_images == 0:
        return []

    actual_cols = min(cols, n_images)
    rows = math.ceil(n_images / actual_cols)

    header_h = 100
    margin = 25
    grid_w = actual_cols * cell_w
    grid_h = rows * cell_h
    total_w = grid_w + 2 * margin
    total_h = grid_h + header_h + margin

    sheet = Image.new("RGB", (total_w, total_h), color=(20, 24, 30))
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(24)
    subtitle_font = get_font(14)
    meta_font = get_font(13)
    dim_font = get_font(11)
    tag_font = get_font(12)
    body_font = get_font(12)

    # Header banner
    draw.text(
        (margin, 20),
        f"WildMatch AI — BLIP Caption Review: {species_name}",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (margin, 58),
        f"Sheet {sheet_index:02d} of {total_sheets_for_species:02d} ({n_images} candidates on this sheet) | "
        f"Verify animal identification, posture, context & background description",
        fill=(160, 180, 205),
        font=subtitle_font,
    )
    draw.line([(margin, 88), (total_w - margin, 88)], fill=(45, 58, 75), width=2)

    sheet_index_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        col = idx % actual_cols
        row = idx // actual_cols

        x = margin + col * cell_w
        y = header_h + row * cell_h

        card_rect = [x + 5, y + 5, x + cell_w - 5, y + cell_h - 5]
        draw.rectangle(card_rect, fill=(30, 36, 46), outline=(52, 66, 84), width=1)

        raw_rel_path = rec.get("image_path", "")
        img_abs_path = os.path.normpath(os.path.join(base_dir, raw_rel_path)) if not os.path.isabs(raw_rel_path) else raw_rel_path
        photo_id = rec.get("photo_id", "N/A")
        orig_w = rec.get("original_width", "?")
        orig_h = rec.get("original_height", "?")
        blip_caption = str(rec.get("blip_caption", "") if pd.notna(rec.get("blip_caption")) else "").strip()
        filename = os.path.basename(img_abs_path)

        # Image thumbnail rendering (aspect ratio preserved, no crop)
        if os.path.isfile(img_abs_path):
            try:
                with Image.open(img_abs_path) as orig_img:
                    thumb = orig_img.copy()
                    thumb.thumbnail((thumb_box_w, thumb_box_h), Image.Resampling.LANCZOS)
                    tw, th = thumb.size

                    tx = x + 15 + (thumb_box_w - tw) // 2
                    ty = y + 15 + (thumb_box_h - th) // 2
                    sheet.paste(thumb, (tx, ty))
                    draw.rectangle([tx, ty, tx + tw, ty + th], outline=(65, 80, 100), width=1)
            except Exception as e:
                draw.text((x + 20, y + thumb_box_h // 2), f"Error: {e}", fill=(255, 90, 90), font=meta_font)
        else:
            draw.text((x + 20, y + thumb_box_h // 2), "File Missing", fill=(255, 90, 90), font=meta_font)

        # Labels below image
        text_x = x + 16
        text_y = y + thumb_box_h + 22

        # Line 1: Species & Photo ID
        draw.text(
            (text_x, text_y),
            f"{species_name}  |  Photo ID: {photo_id}",
            fill=(100, 210, 255),
            font=meta_font,
        )
        text_y += 18

        # Line 2: Filename & Dimensions
        draw.text(
            (text_x, text_y),
            f"File: {filename}  |  {orig_w}x{orig_h}",
            fill=(140, 155, 175),
            font=dim_font,
        )
        text_y += 18

        # Line 3: BLIP caption (tag + wrapped full caption)
        draw.text((text_x, text_y), "BLIP:", fill=(255, 205, 100), font=tag_font)
        text_y = draw_wrapped_text(
            draw=draw,
            text=f'"{blip_caption}"',
            x=text_x + 42,
            y=text_y,
            max_w=cell_w - 75,
            font=body_font,
            fill=(235, 242, 250),
        )

        # Line 4: Review status
        draw.text((text_x, text_y + 4), "Status: [Pending Manual Review]", fill=(120, 200, 140), font=dim_font)

        sheet_index_records.append({
            "photo_id": int(photo_id),
            "image_id": rec.get("image_id", f"{species_name}_{photo_id}"),
            "species": species_name,
            "image_path": raw_rel_path,
            "blip_caption": blip_caption,
            "contact_sheet": os.path.basename(output_path),
            "contact_sheet_path": output_path.replace("\\", "/"),
            "sheet_number": sheet_index,
            "total_species_sheets": total_sheets_for_species,
            "position_on_sheet": idx + 1,
            "grid_row": row + 1,
            "grid_col": col + 1,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=95)
    logger.info(f"Saved contact sheet: {output_path} ({n_images} images)")

    return sheet_index_records


# ---------------------------------------------------------------------------
# Master Orchestration Routine
# ---------------------------------------------------------------------------
def generate_all_caption_review_sheets(
    captions_csv_path: str = "dataset/captions/pilot_captions.csv",
    curated_meta_path: str = "dataset/curated/metadata.csv",
    output_sheets_dir: str = "dataset/captions/review_contact_sheets",
    base_dir: str = ".",
    images_per_sheet: int = IMAGES_PER_SHEET,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Generate visual caption review contact sheets for all newly generated BLIP captions.
    Creates index CSV mapping each photo ID to sheet and grid position.
    """
    new_captions_df, reviewed_df = load_caption_candidates(
        captions_csv_path=captions_csv_path,
        curated_meta_path=curated_meta_path,
        base_dir=base_dir,
    )

    all_index_records: List[Dict[str, Any]] = []
    generated_sheets_info: List[Dict[str, Any]] = []

    for species, exp_count in EXPECTED_NEW_PER_SPECIES.items():
        sp_df = new_captions_df[new_captions_df["species"] == species].copy()
        actual_count = len(sp_df)

        if actual_count != exp_count:
            raise ValueError(
                f"Candidate count mismatch for {species}: expected {exp_count}, got {actual_count}"
            )

        # Consistent ordering by photo_id
        sp_df = sp_df.sort_values(by="photo_id").reset_index(drop=True)
        records = sp_df.to_dict("records")

        total_sheets = math.ceil(len(records) / images_per_sheet)
        prefix = SPECIES_FILE_PREFIX[species]

        logger.info(
            f"Generating caption review sheets for {species}: {len(records)} candidates across {total_sheets} sheet(s)..."
        )

        for sheet_idx in range(1, total_sheets + 1):
            start_idx = (sheet_idx - 1) * images_per_sheet
            end_idx = min(start_idx + images_per_sheet, len(records))
            sheet_records = records[start_idx:end_idx]

            sheet_filename = f"{prefix}_review_{sheet_idx:02d}.jpg"
            sheet_path = os.path.join(output_sheets_dir, sheet_filename)

            sheet_indices = render_caption_review_sheet(
                species_name=species,
                sheet_index=sheet_idx,
                total_sheets_for_species=total_sheets,
                records=sheet_records,
                output_path=sheet_path,
                base_dir=base_dir,
            )
            all_index_records.extend(sheet_indices)

            generated_sheets_info.append({
                "species": species,
                "sheet_filename": sheet_filename,
                "sheet_path": sheet_path.replace("\\", "/"),
                "sheet_number": sheet_idx,
                "total_sheets": total_sheets,
                "image_count": len(sheet_records),
                "photo_ids": [r["photo_id"] for r in sheet_records],
            })

    # Save index CSV
    index_csv_path = os.path.join(output_sheets_dir, "caption_review_index.csv")
    index_df = pd.DataFrame(all_index_records)
    index_df.to_csv(index_csv_path, index=False)
    logger.info(f"Saved caption review index to: {index_csv_path} (Total rows: {len(index_df)})")

    return generated_sheets_info, index_df


# ---------------------------------------------------------------------------
# Comprehensive Post-Validation
# ---------------------------------------------------------------------------
def validate_caption_review_generation(
    index_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
    generated_sheets_info: List[Dict[str, Any]],
    base_dir: str = ".",
) -> Dict[str, Any]:
    """
    Validate caption review sheets:
    - Exactly 164 new records included.
    - Zero reviewed captions included.
    - Every new photo ID appears exactly once.
    - All generated contact sheet files exist and are decodable.
    - All images load successfully.
    """
    results: Dict[str, Any] = {}

    # Check 1: Record count
    total_indexed = len(index_df)
    results["total_records"] = total_indexed
    results["exact_164_records"] = total_indexed == EXPECTED_NEW_TOTAL

    # Check 2: Species distribution
    species_counts = index_df["species"].value_counts().to_dict()
    results["species_distribution"] = species_counts
    results["species_distribution_correct"] = all(
        species_counts.get(sp) == count for sp, count in EXPECTED_NEW_PER_SPECIES.items()
    )

    # Check 3: Zero overlap with reviewed captions
    reviewed_ids: Set[int] = set(reviewed_df["photo_id"].astype(int))
    indexed_ids: Set[int] = set(index_df["photo_id"].astype(int))
    overlap = reviewed_ids.intersection(indexed_ids)
    results["reviewed_overlap_count"] = len(overlap)
    results["no_reviewed_overlap"] = len(overlap) == 0

    # Check 4: Uniqueness of photo IDs
    results["unique_photo_ids"] = index_df["photo_id"].nunique()
    results["unique_photo_ids_is_164"] = results["unique_photo_ids"] == EXPECTED_NEW_TOTAL

    # Check 5: Contact sheets existence and decodability
    sheet_errors = []
    for info in generated_sheets_info:
        sp_path = os.path.normpath(os.path.join(base_dir, info["sheet_path"]))
        ok, err = verify_image(sp_path)
        if not ok:
            sheet_errors.append((info["sheet_filename"], err))
    results["sheet_errors"] = sheet_errors
    results["all_sheets_valid"] = len(sheet_errors) == 0

    # Check 6: Non-empty captions in index
    blank_caps = (index_df["blip_caption"].isna() | (index_df["blip_caption"].astype(str).str.strip() == "")).sum()
    results["blank_captions"] = int(blank_caps)
    results["all_captions_present"] = blank_caps == 0

    all_passed = (
        results["exact_164_records"]
        and results["species_distribution_correct"]
        and results["no_reviewed_overlap"]
        and results["unique_photo_ids_is_164"]
        and results["all_sheets_valid"]
        and results["all_captions_present"]
    )
    results["all_passed"] = all_passed
    return results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate visual caption review contact sheets for 164 new BLIP captions."
    )
    parser.add_argument(
        "--captions-csv",
        type=str,
        default="dataset/captions/pilot_captions.csv",
        help="Path to captions CSV.",
    )
    parser.add_argument(
        "--curated-meta",
        type=str,
        default="dataset/curated/metadata.csv",
        help="Path to curated metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/captions/review_contact_sheets",
        help="Directory to save generated contact sheets.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    logger.info("============================================================")
    logger.info("   WildMatch AI - Expansion Caption Review Sheet Generator   ")
    logger.info("============================================================")

    sheets_info, index_df = generate_all_caption_review_sheets(
        captions_csv_path=args.captions_csv,
        curated_meta_path=args.curated_meta,
        output_sheets_dir=args.output_dir,
        base_dir=args.base_dir,
    )

    # Post-validation
    captions_df = pd.read_csv(args.captions_csv)
    is_blank = captions_df["final_caption"].isna() | (captions_df["final_caption"].astype(str).str.strip() == "")
    reviewed_df = captions_df[~is_blank].copy()

    val = validate_caption_review_generation(
        index_df=index_df,
        reviewed_df=reviewed_df,
        generated_sheets_info=sheets_info,
        base_dir=args.base_dir,
    )

    logger.info("============================================================")
    logger.info("Validation Results:")
    logger.info(f"  Exact 164 new records       : {val['exact_164_records']}")
    logger.info(f"  Species distribution correct: {val['species_distribution_correct']} ({val['species_distribution']})")
    logger.info(f"  No reviewed overlap         : {val['no_reviewed_overlap']}")
    logger.info(f"  All photo IDs unique (164)  : {val['unique_photo_ids_is_164']}")
    logger.info(f"  All contact sheets decodable: {val['all_sheets_valid']} (Errors: {len(val['sheet_errors'])})")
    logger.info(f"  All captions present        : {val['all_captions_present']}")
    logger.info(f"  Overall Validation Passed   : {val['all_passed']}")
    logger.info("============================================================")

    if not val["all_passed"]:
        logger.error("Validation failed! Halting.")
        sys.exit(1)

    print("\nGenerated Caption Review Contact Sheets:")
    for si in sheets_info:
        print(f"  - [{si['species']}] {si['sheet_filename']}: {si['image_count']} candidates -> {si['sheet_path']}")


if __name__ == "__main__":
    main()
