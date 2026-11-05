"""
WildMatch AI - Stage 2: BLIP-based Image Captioning

This module handles:
1. Environment and hardware inspection (CUDA/CPU, RAM, VRAM).
2. Loading curated metadata from `dataset/curated/metadata.csv`.
3. Generating descriptive captions using `Salesforce/blip-image-captioning-base`.
4. Skipping already captioned images and preserving existing manual `final_caption` reviews.
5. Saving all results to `dataset/captions/pilot_captions.csv` with safe rerun handling.
"""

# Import torch first to avoid Windows OpenMP/MKL DLL conflict with numpy/pandas
import torch

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import psutil
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_Captioner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"

CAPTION_COLUMNS = [
    "image_id",
    "photo_id",
    "species",
    "image_path",
    "blip_caption",
    "final_caption",
    "model_name",
]


# ---------------------------------------------------------------------------
# Hardware & Environment Inspection
# ---------------------------------------------------------------------------

def inspect_environment() -> Dict[str, Any]:
    """Inspect system RAM, PyTorch version, and CUDA/GPU availability."""
    ram = psutil.virtual_memory()
    total_ram_gb = round(ram.total / (1024 ** 3), 2)
    avail_ram_gb = round(ram.available / (1024 ** 3), 2)

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    gpu_info = {}
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        allocated_vram_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        gpu_info = {
            "device_count": gpu_count,
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "allocated_vram_gb": allocated_vram_gb,
        }

    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": avail_ram_gb,
        "cuda_available": cuda_available,
        "device": device,
        "gpu_info": gpu_info,
    }

    logger.info("============================================================")
    logger.info("                HARDWARE & RUNTIME INSPECTION               ")
    logger.info("============================================================")
    logger.info(f"Python:            {info['python_version']}")
    logger.info(f"PyTorch:           {info['torch_version']}")
    logger.info(f"System RAM:        {avail_ram_gb} GB available / {total_ram_gb} GB total")
    logger.info(f"CUDA Available:    {cuda_available}")
    if cuda_available:
        logger.info(f"GPU Name:          {gpu_info['gpu_name']}")
        logger.info(f"VRAM:              {gpu_info['total_vram_gb']} GB total")
    else:
        logger.info("Device Selected:   CPU (No CUDA GPU detected)")
    logger.info("============================================================")

    return info


# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------

def load_curated_records(
    curated_metadata_path: str,
    base_dir: str,
    pilot_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load curated image records from dataset/curated/metadata.csv.
    Validates that each image file exists and can be opened with Pillow.
    If pilot_only=True, selects exactly 1 image per species.
    """
    if not os.path.exists(curated_metadata_path):
        raise FileNotFoundError(f"Curated metadata not found at: {curated_metadata_path}")

    df = pd.read_csv(curated_metadata_path)
    logger.info(f"Loaded {len(df)} curated records from: {curated_metadata_path}")

    if pilot_only:
        species_list = ["Tiger", "Lion", "Asian Elephant", "Chital"]
        selected_records: List[Dict[str, Any]] = []
        for sp in species_list:
            sp_rows = df[df["species"] == sp]
            for _, row in sp_rows.iterrows():
                rel_path = str(row["image_path"])
                abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
                if os.path.isfile(abs_path):
                    try:
                        with Image.open(abs_path) as img:
                            img.load()
                        rec = row.to_dict()
                        rec["abs_image_path"] = abs_path
                        selected_records.append(rec)
                        break
                    except Exception:
                        continue
        return selected_records

    # Full curated dataset mode (all 36 images)
    valid_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rel_path = str(row["image_path"])
        abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
        if not os.path.isfile(abs_path):
            logger.warning(f"File missing on disk: {abs_path}")
            continue
        try:
            with Image.open(abs_path) as img:
                img.load()
            rec = row.to_dict()
            rec["abs_image_path"] = abs_path
            valid_records.append(rec)
        except Exception as e:
            logger.warning(f"Image {abs_path} failed to decode: {e}")

    logger.info(f"Verified {len(valid_records)} valid curated images on disk.")
    return valid_records


# ---------------------------------------------------------------------------
# BLIP Caption Generation
# ---------------------------------------------------------------------------

def generate_captions(
    model_name: str = DEFAULT_MODEL_NAME,
    base_dir: str = ".",
    output_csv_path: Optional[str] = None,
    pilot_only: bool = False,
) -> pd.DataFrame:
    """
    Load BLIP model, generate descriptive captions for curated images,
    skipping already captioned images and preserving manual final_caption values.
    """
    base_dir = os.path.abspath(base_dir)
    curated_meta_path = os.path.join(base_dir, "dataset", "curated", "metadata.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(base_dir, "dataset", "captions", "pilot_captions.csv")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 1. Hardware Inspection
    hw_info = inspect_environment()
    device = hw_info["device"]

    # 2. Curated Image Loading
    curated_records = load_curated_records(curated_meta_path, base_dir, pilot_only=pilot_only)

    # 3. Load Existing Captions to Preserve Reviews & Skip Already Processed
    existing_captions_map: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(output_csv_path):
        try:
            prev_df = pd.read_csv(output_csv_path)
            for _, r in prev_df.iterrows():
                p_id = r.get("photo_id")
                if pd.notna(p_id) and str(p_id).isdigit():
                    int_id = int(p_id)
                    existing_captions_map[int_id] = {
                        "image_id": str(r.get("image_id", "")),
                        "photo_id": int_id,
                        "species": str(r.get("species", "")),
                        "image_path": str(r.get("image_path", "")),
                        "blip_caption": str(r.get("blip_caption", "") if pd.notna(r.get("blip_caption")) else ""),
                        "final_caption": str(r.get("final_caption", "") if pd.notna(r.get("final_caption")) else ""),
                        "model_name": str(r.get("model_name", model_name)),
                    }
            logger.info(f"Loaded {len(existing_captions_map)} existing caption records from: {output_csv_path}")
        except Exception as e:
            logger.warning(f"Could not read existing caption file ({e}). Starting fresh.")

    # Identify images needing captioning vs already captioned
    to_process: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    for rec in curated_records:
        photo_id = int(rec["photo_id"])
        if photo_id in existing_captions_map and existing_captions_map[photo_id]["blip_caption"]:
            skipped_records.append(existing_captions_map[photo_id])
        else:
            to_process.append(rec)

    logger.info(f"Total curated images:    {len(curated_records)}")
    logger.info(f"Already captioned (skip): {len(skipped_records)}")
    logger.info(f"New images to process:    {len(to_process)}")

    output_rows: List[Dict[str, Any]] = list(skipped_records)
    failed_images: List[Tuple[str, str]] = []

    if to_process:
        # 4. Model Loading
        logger.info(f"Loading BLIP model and processor: {model_name}...")
        start_load = time.time()
        try:
            processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
            model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True).to(device)
        except Exception:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        logger.info(f"Model loaded successfully in {round(time.time() - start_load, 2)}s.")

        # 5. Sequential Caption Generation on CPU
        total_to_process = len(to_process)
        for idx, rec in enumerate(to_process, start=1):
            photo_id = int(rec["photo_id"])
            image_id = str(rec["image_id"])
            species = str(rec["species"])
            image_path = str(rec["image_path"])
            abs_path = rec["abs_image_path"]

            logger.info(f"[{idx}/{total_to_process}] Generating caption for {species} ({image_id})...")
            t0 = time.time()

            try:
                with Image.open(abs_path) as raw_img:
                    rgb_image = raw_img.convert("RGB")
                    inputs = processor(rgb_image, return_tensors="pt").to(device)

                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=5,
                            early_stopping=True,
                        )
                    generated_caption = processor.decode(out[0], skip_special_tokens=True).strip()

                gen_time = round(time.time() - t0, 2)
                logger.info(f" -> '{generated_caption}' ({gen_time}s)")

                # If existing record had a final_caption, preserve it; otherwise leave blank
                existing_final = ""
                if photo_id in existing_captions_map:
                    existing_final = existing_captions_map[photo_id].get("final_caption", "")

                entry = {
                    "image_id": image_id,
                    "photo_id": photo_id,
                    "species": species,
                    "image_path": image_path,
                    "blip_caption": generated_caption,
                    "final_caption": existing_final,
                    "model_name": model_name,
                }
                output_rows.append(entry)

                # Periodic save every 5 images for safety
                if idx % 5 == 0 or idx == total_to_process:
                    temp_df = pd.DataFrame(output_rows)[CAPTION_COLUMNS].drop_duplicates(subset=["photo_id"], keep="last")
                    temp_df.to_csv(output_csv_path, index=False)
                    logger.info(f"Checkpoint saved ({len(temp_df)} records) to {output_csv_path}")

            except Exception as e:
                logger.error(f"Failed to generate caption for {image_id} ({abs_path}): {e}")
                failed_images.append((image_id, str(e)))
            finally:
                if "rgb_image" in locals():
                    del rgb_image
                if "inputs" in locals():
                    del inputs
                if "out" in locals():
                    del out
                gc.collect()
    else:
        logger.info("All images are already captioned. No new captions needed.")

    # 6. Final Save CSV
    results_df = pd.DataFrame(output_rows)[CAPTION_COLUMNS]
    results_df = results_df.drop_duplicates(subset=["photo_id"], keep="last")

    # Safety backup before saving final captions
    if os.path.exists(output_csv_path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_csv_path}.bak_{ts}"
        shutil.copy2(output_csv_path, backup_path)
        logger.info(f"Created safety backup of captions at: {backup_path}")

    results_df.to_csv(output_csv_path, index=False)
    logger.info(f"Final captions saved: {output_csv_path} (Total rows: {len(results_df)})")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    logger.info("                  CAPTION GENERATION SUMMARY                ")
    logger.info("=" * 60)
    logger.info(f"Total curated images:      {len(curated_records)}")
    logger.info(f"Skipped (already present): {len(skipped_records)}")
    logger.info(f"Processed in this run:     {len(to_process)}")
    logger.info(f"Failed:                    {len(failed_images)}")
    logger.info(f"Total records in CSV:      {len(results_df)}")
    logger.info("-" * 60)
    logger.info("Caption count per species:")
    for sp, cnt in results_df["species"].value_counts().items():
        logger.info(f"  - {sp:18s}: {cnt}")
    logger.info("=" * 60)

    if failed_images:
        logger.warning(f"Failed images ({len(failed_images)}): {failed_images}")

    return results_df


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WildMatch AI - Stage 2: BLIP Image Captioning"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Pretrained model name or path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save captions CSV (default: dataset/captions/pilot_captions.csv)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Run pilot mode on 4 representative images (default: False)",
    )

    args = parser.parse_args()

    generate_captions(
        model_name=args.model_name,
        base_dir=args.base_dir,
        output_csv_path=args.output_csv,
        pilot_only=args.pilot,
    )


if __name__ == "__main__":
    main()
