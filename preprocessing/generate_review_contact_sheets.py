"""
WildMatch AI - Visual Curation Review Contact Sheet Generator

Generates visual contact sheets for the remaining 28 curated images to enable
final visual verification of BLIP generated captions against curated photos.
"""

import math
import os
import sys
import textwrap
from typing import Any, Dict, List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Reusing helper utilities from curate_dataset without modifying existing scripts
sys.path.insert(0, os.path.abspath("."))
from preprocessing.curate_dataset import get_font, resolve_path, verify_image

# ---------------------------------------------------------------------------
# Constants & Exclusion List
# ---------------------------------------------------------------------------
EXCLUDED_PHOTO_IDS = {
    759553570,  # Tiger (pilot reviewed)
    756403166,  # Lion (pilot reviewed)
    754925769,  # Asian Elephant (pilot reviewed)
    757904093,  # Chital (pilot reviewed)
    754466576,  # Asian Elephant (initial inspection)
    750817521,  # Asian Elephant (initial inspection)
    746410908,  # Chital (initial inspection)
    756589825,  # Chital (initial inspection)
}

SPECIES_FILE_MAP = {
    "Tiger": "tiger_review.jpg",
    "Lion": "lion_review.jpg",
    "Asian Elephant": "elephant_review.jpg",
    "Chital": "chital_review.jpg",
}


# ---------------------------------------------------------------------------
# Text Wrapping Helper
# ---------------------------------------------------------------------------

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

    lines = []
    current_line = []

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
# Contact Sheet Rendering
# ---------------------------------------------------------------------------

def generate_review_sheet(
    species_name: str,
    records: List[Dict[str, Any]],
    base_dir: str,
    output_path: str,
    cols: int = 4,
    cell_w: int = 440,
    cell_h: int = 470,
    thumb_box_w: int = 410,
    thumb_box_h: int = 270,
) -> None:
    """
    Generate a visual verification contact sheet for a species.
    Displays the image without cropping, species, photo ID, BLIP caption,
    and final caption (or pending status).
    """
    n_images = len(records)
    if n_images == 0:
        return

    cols = min(cols, n_images)
    rows = math.ceil(n_images / cols)

    header_h = 100
    margin = 30
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    total_w = max(grid_w + 2 * margin, 700)
    total_h = grid_h + header_h + margin

    sheet = Image.new("RGB", (total_w, total_h), color=(20, 24, 30))
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(26)
    sub_font = get_font(14)
    meta_font = get_font(13)
    dim_font = get_font(11)
    caption_tag_font = get_font(12)
    caption_body_font = get_font(12)

    # Header banner
    draw.text(
        (margin, 22),
        f"WildMatch AI — Caption Review: {species_name}",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (margin, 60),
        f"Displaying {n_images} image(s) for visual caption verification | Excludes previously verified pilot images",
        fill=(160, 180, 205),
        font=sub_font,
    )
    draw.line([(margin, 88), (total_w - margin, 88)], fill=(50, 65, 85), width=2)

    for idx, rec in enumerate(records):
        col = idx % cols
        row = idx // cols

        x = margin + col * cell_w
        y = header_h + row * cell_h

        card_rect = [x + 6, y + 6, x + cell_w - 6, y + cell_h - 6]
        draw.rectangle(card_rect, fill=(30, 36, 46), outline=(55, 70, 90), width=1)

        img_path = resolve_path(base_dir, rec.get("image_path", ""))
        photo_id = rec.get("photo_id", "N/A")
        blip_caption = str(rec.get("blip_caption", "") if pd.notna(rec.get("blip_caption")) else "").strip()
        final_caption = str(rec.get("final_caption", "") if pd.notna(rec.get("final_caption")) else "").strip()
        filename = os.path.basename(img_path)

        # Image thumbnail rendering (preserve aspect ratio, no crop)
        if os.path.isfile(img_path):
            try:
                with Image.open(img_path) as orig_img:
                    thumb = orig_img.copy()
                    thumb.thumbnail((thumb_box_w, thumb_box_h), Image.Resampling.LANCZOS)
                    tw, th = thumb.size

                    tx = x + 15 + (thumb_box_w - tw) // 2
                    ty = y + 15 + (thumb_box_h - th) // 2
                    sheet.paste(thumb, (tx, ty))
                    draw.rectangle([tx, ty, tx + tw, ty + th], outline=(70, 85, 105), width=1)
            except Exception as e:
                draw.text((x + 20, y + 120), f"Error loading image: {e}", fill=(255, 100, 100), font=meta_font)
        else:
            draw.text((x + 20, y + 120), "Image file missing", fill=(255, 100, 100), font=meta_font)

        # Metadata text
        text_x = x + 18
        text_y = y + thumb_box_h + 24
        draw.text(
            (text_x, text_y),
            f"{species_name}  |  Photo ID: {photo_id}",
            fill=(100, 210, 255),
            font=meta_font,
        )
        text_y += 18
        draw.text((text_x, text_y), f"File: {filename}", fill=(130, 145, 165), font=dim_font)
        text_y += 18

        # BLIP caption (gold/yellow badge)
        draw.text((text_x, text_y), "BLIP:", fill=(255, 205, 100), font=caption_tag_font)
        text_y = draw_wrapped_text(
            draw=draw,
            text=f'"{blip_caption}"',
            x=text_x + 40,
            y=text_y,
            max_w=cell_w - 75,
            font=caption_body_font,
            fill=(235, 240, 250),
        )

        # Final caption (green if reviewed, muted orange if pending)
        text_y += 4
        if final_caption:
            draw.text((text_x, text_y), "Final:", fill=(110, 240, 150), font=caption_tag_font)
            draw_wrapped_text(
                draw=draw,
                text=f'"{final_caption}"',
                x=text_x + 40,
                y=text_y,
                max_w=cell_w - 75,
                font=caption_body_font,
                fill=(210, 255, 225),
            )
        else:
            draw.text((text_x, text_y), "Final:", fill=(200, 140, 110), font=caption_tag_font)
            draw.text((text_x + 40, text_y), "[Pending Review]", fill=(175, 130, 110), font=caption_body_font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=95)
    print(f"Saved contact sheet for {species_name:15s} ({n_images:2d} images) -> {output_path}")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.abspath(".")
    captions_csv = os.path.join(base_dir, "dataset", "captions", "pilot_captions.csv")
    curated_meta_csv = os.path.join(base_dir, "dataset", "curated", "metadata.csv")
    out_dir = os.path.join(base_dir, "dataset", "captions", "review_contact_sheets")

    if not os.path.isfile(captions_csv):
        print(f"Error: {captions_csv} not found.")
        sys.exit(1)
    if not os.path.isfile(curated_meta_csv):
        print(f"Error: {curated_meta_csv} not found.")
        sys.exit(1)

    captions_df = pd.read_csv(captions_csv)
    curated_df = pd.read_csv(curated_meta_csv)

    print("=" * 65)
    print("      WildMatch AI — Review Contact Sheet Generator       ")
    print("=" * 65)
    print(f"Total curated records:            {len(curated_df)}")
    print(f"Total captioned records:          {len(captions_df)}")
    print(f"Excluded previously reviewed IDs: {len(EXCLUDED_PHOTO_IDS)}")

    # Filter out excluded photo IDs
    remaining_df = captions_df[~captions_df["photo_id"].isin(EXCLUDED_PHOTO_IDS)].copy()
    # Sort for consistent ordering
    remaining_df = remaining_df.sort_values(by=["species", "photo_id"])

    total_remaining = len(remaining_df)
    unique_remaining = remaining_df["photo_id"].nunique()
    print(f"Remaining images for review:      {total_remaining} (unique: {unique_remaining})")
    print("-" * 65)

    if total_remaining != 28:
        print(f"WARNING: Expected exactly 28 images, but found {total_remaining}.")
    if unique_remaining != total_remaining:
        print(f"ERROR: Duplicate photo IDs detected in remaining images!")
        sys.exit(1)

    # Verify all remaining image files
    missing_or_corrupt = []
    for _, r in remaining_df.iterrows():
        p = resolve_path(base_dir, r["image_path"])
        ok, err = verify_image(p)
        if not ok:
            missing_or_corrupt.append((r["photo_id"], p, err))

    if missing_or_corrupt:
        print(f"ERROR: Found {len(missing_or_corrupt)} unreadable images!")
        for pid, p, err in missing_or_corrupt:
            print(f"  - Photo {pid}: {p} ({err})")
        sys.exit(1)
    else:
        print("Image integrity verification:     All 28 images exist and decode as valid RGB JPEG.")

    print("-" * 65)
    print("Remaining images per species:")
    species_counts = remaining_df["species"].value_counts()
    for sp, cnt in species_counts.items():
        print(f"  - {sp:18s}: {cnt:2d} image(s)")
    print("-" * 65)

    # Generate sheets per species
    created_files = []
    # Grid column configuration by species size
    cols_map = {
        "Tiger": 4,          # 16 images -> 4x4
        "Lion": 3,           # 9 images  -> 3x3
        "Asian Elephant": 1, # 1 image   -> 1x1
        "Chital": 2,         # 2 images  -> 2x1
    }

    for sp, out_name in SPECIES_FILE_MAP.items():
        sp_records = remaining_df[remaining_df["species"] == sp].to_dict("records")
        out_path = os.path.join(out_dir, out_name)
        cols = cols_map.get(sp, 3)
        generate_review_sheet(
            species_name=sp,
            records=sp_records,
            base_dir=base_dir,
            output_path=out_path,
            cols=cols,
        )
        created_files.append((sp, out_path, len(sp_records)))

    print("=" * 65)
    print("All review contact sheets generated successfully!")
    print(f"Saved in: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
