"""
WildMatch AI - Image Preprocessing Module

This module implements aspect-preserving image preprocessing for the custom CNN:
1. Safe image loading using Pillow (with EXIF orientation correction and RGB conversion).
2. Aspect-preserving resizing to fit within target dimensions (default 224 x 224).
3. Symmetric zero-padding (letterboxing/pillarboxing) to create square images without distortion.
4. PyTorch tensor conversion:
   - Channels-first format: [3, 224, 224] (C x H x W).
   - Data type: torch.float32.
   - Value range: [0.0, 1.0].
5. Validation utilities for testing all curated dataset images.

Design Notes:
- CNN-specific normalization (e.g. mean/std scaling) is intentionally deferred to
  the dataset/DataLoader stage so that raw [0.0, 1.0] tensor values are preserved here.
- CLIP's pretrained vision encoder uses its dedicated Hugging Face processor
  (CLIPProcessor) during inference and training and is kept completely independent.
- Curated and raw images are strictly read-only and never modified or overwritten.
"""

# Import torch first to avoid Windows OpenMP/MKL DLL initialization conflicts with numpy/pandas
import torch

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_ImagePreprocessing")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SIZE: int = 224
TARGET_CHANNELS: int = 3
TARGET_SHAPE: Tuple[int, int, int] = (TARGET_CHANNELS, TARGET_SIZE, TARGET_SIZE)
DEFAULT_PAD_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Neutral zero-padding (black)

EXPECTED_RECORD_COUNT: int = 36
EXPECTED_SPECIES: Tuple[str, ...] = ("Tiger", "Lion", "Asian Elephant", "Chital")


# ---------------------------------------------------------------------------
# Core Image Preprocessing Functions
# ---------------------------------------------------------------------------
def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Safely load an image from disk using Pillow:
    - Verifies file existence.
    - Applies EXIF orientation correction (ImageOps.exif_transpose).
    - Enforces RGB colour mode (converting RGBA, grayscale, or palette images).
    - Loads pixel data into memory so file handles can be closed safely.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Pillow Image in RGB format.

    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If Pillow cannot decode the file.
    """
    path_obj = Path(image_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(path_obj) as img:
        # Correct orientation based on EXIF metadata if present
        img_transposed = ImageOps.exif_transpose(img)
        if img_transposed is None:
            img_transposed = img

        # Ensure RGB mode (handles RGBA, grayscale, or palette images)
        if img_transposed.mode != "RGB":
            img_rgb = img_transposed.convert("RGB")
        else:
            img_rgb = img_transposed.copy()

        # Force loading of image data into memory
        img_rgb.load()
        return img_rgb


def resize_and_pad(
    image: Image.Image,
    target_size: int = TARGET_SIZE,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Resize an image to fit inside a target square while preserving its aspect ratio,
    then pad the remaining canvas with a uniform colour to produce a square image.

    Padding Strategy:
    - Scaling factor is calculated as `scale = min(target_size / width, target_size / height)`.
    - Resized image preserves the original aspect ratio without non-uniform stretching or cropping.
    - Image is centered symmetrically within the (target_size x target_size) canvas.
    - Remaining borders (letterbox for landscape, pillarbox for portrait) are padded with `pad_color`.
      The default (0, 0, 0) corresponds to 0.0 in [0, 1] tensor space, providing a standard,
      clean zero-padding baseline for CNN convolution layers.

    Args:
        image: Source Pillow Image in RGB.
        target_size: Desired square canvas size in pixels (default: 224).
        pad_color: RGB tuple for background padding (default: (0, 0, 0)).
        resample: Resampling filter for high-quality downscaling (default: LANCZOS).

    Returns:
        A tuple of:
          - Padded square Pillow Image of size (target_size, target_size).
          - Resized dimensions (new_w, new_h).
          - Padding offsets (pad_left, pad_top).
    """
    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    # Compute scale factor preserving aspect ratio
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # High quality resize
    resized_image = image.resize((new_w, new_h), resample=resample)

    # Create new square canvas with uniform pad_color
    canvas = Image.new("RGB", (target_size, target_size), pad_color)

    # Compute symmetric center offsets
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    # Paste resized image onto center of canvas
    canvas.paste(resized_image, (pad_left, pad_top))

    return canvas, (new_w, new_h), (pad_left, pad_top)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a Pillow RGB image into a PyTorch float32 tensor:
    - Channels-first memory layout: [3, H, W] (C x H x W).
    - Data type: torch.float32.
    - Normalized pixel value range: [0.0, 1.0] (divided by 255.0).

    Args:
        image: Pillow Image in RGB mode.

    Returns:
        torch.Tensor of shape [3, H, W], dtype torch.float32, values in [0.0, 1.0].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert PIL Image to numpy float32 array in [H, W, C]
    np_img = np.array(image, dtype=np.float32)

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    np_img = np_img / 255.0

    # Convert to PyTorch tensor and permute dimensions from [H, W, C] to [C, H, W]
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
    return tensor


def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    target_size: int = TARGET_SIZE,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
) -> torch.Tensor:
    """
    High-level end-to-end preprocessing function:
    1. Loads or accepts an image and ensures RGB format.
    2. Resizes while preserving aspect ratio and pads to (target_size x target_size).
    3. Converts to a PyTorch tensor: [3, target_size, target_size], float32, range [0, 1].

    Args:
        image_input: File path (str or Path) OR an existing Pillow Image.
        target_size: Target square dimension in pixels (default: 224).
        pad_color: RGB padding colour tuple (default: (0, 0, 0)).

    Returns:
        PyTorch tensor of shape [3, target_size, target_size], dtype torch.float32.
    """
    if isinstance(image_input, (str, Path)):
        img = load_image(image_input)
    elif isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":
            img = image_input.convert("RGB")
        else:
            img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    padded_img, _, _ = resize_and_pad(img, target_size=target_size, pad_color=pad_color)
    tensor = image_to_tensor(padded_img)
    return tensor


# ---------------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------------
def validate_curated_images(
    metadata_csv: Union[str, Path] = "dataset/curated/metadata.csv",
    save_samples_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Validate image preprocessing across all curated images:
    - Checks file existence and decodability for all 36 records.
    - Confirms correct RGB conversion, resizing, and padding.
    - Verifies tensor shape [3, 224, 224], float32 type, and value range [0, 1].
    - Validates species coverage and aspect ratio distribution.
    - Optionally saves sample preprocessed images for visual inspection in `save_samples_dir`.

    Args:
        metadata_csv: Path to `dataset/curated/metadata.csv`.
        save_samples_dir: Optional directory to save test/sample preprocessed images.

    Returns:
        Dictionary containing validation results and summary metrics.
    """
    meta_path = Path(metadata_csv)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Curated metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(meta_path)
    total_records = len(df)

    if save_samples_dir:
        samples_dir_path = Path(save_samples_dir)
        samples_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        samples_dir_path = None

    results: Dict[str, Any] = {
        "total_records": total_records,
        "unique_photos": df["photo_id"].nunique(),
        "species_counts": df["species"].value_counts().to_dict(),
        "processed_success": 0,
        "failed_records": [],
        "aspect_ratios": {"landscape": 0, "portrait": 0, "square": 0},
        "sample_records": {},
    }

    # Track one representative sample per species for reporting
    species_seen = set()

    for idx, row in df.iterrows():
        photo_id = row["photo_id"]
        species = row["species"]
        image_path = row["image_path"]

        try:
            # 1. Load image safely
            img = load_image(image_path)
            orig_w, orig_h = img.size

            # Aspect ratio classification
            if orig_w > orig_h:
                results["aspect_ratios"]["landscape"] += 1
            elif orig_h > orig_w:
                results["aspect_ratios"]["portrait"] += 1
            else:
                results["aspect_ratios"]["square"] += 1

            # 2. Resize and pad
            padded_img, (new_w, new_h), (pad_left, pad_top) = resize_and_pad(img)

            # 3. Convert to tensor
            tensor = image_to_tensor(padded_img)

            # 4. Strict assertions
            assert padded_img.size == (TARGET_SIZE, TARGET_SIZE), f"Invalid padded image size: {padded_img.size}"
            assert tensor.shape == torch.Size(TARGET_SHAPE), f"Invalid tensor shape: {tensor.shape}"
            assert tensor.dtype == torch.float32, f"Invalid tensor dtype: {tensor.dtype}"
            assert tensor.min().item() >= 0.0, f"Tensor min value < 0.0: {tensor.min().item()}"
            assert tensor.max().item() <= 1.0, f"Tensor max value > 1.0: {tensor.max().item()}"

            results["processed_success"] += 1

            # Save representative sample (one per species)
            if species not in species_seen:
                species_seen.add(species)
                sample_info = {
                    "photo_id": photo_id,
                    "species": species,
                    "image_path": image_path,
                    "orig_dimensions": f"{orig_w}x{orig_h}",
                    "aspect_ratio": f"{orig_w / orig_h:.2f}:1",
                    "resized_dimensions": f"{new_w}x{new_h}",
                    "padding_offsets": f"left={pad_left}, top={pad_top}",
                    "tensor_shape": list(tensor.shape),
                    "tensor_dtype": str(tensor.dtype),
                    "pixel_range": f"[{tensor.min().item():.3f}, {tensor.max().item():.3f}]",
                }

                if samples_dir_path is not None:
                    sample_file = samples_dir_path / f"sample_{species.lower().replace(' ', '_')}_{photo_id}.jpg"
                    padded_img.save(sample_file, format="JPEG", quality=95)
                    sample_info["sample_file"] = str(sample_file)

                results["sample_records"][species] = sample_info

        except Exception as e:
            results["failed_records"].append({"photo_id": photo_id, "species": species, "error": str(e)})

    # Final summary checks
    results["all_records_accounted"] = results["processed_success"] == EXPECTED_RECORD_COUNT
    results["all_species_represented"] = set(EXPECTED_SPECIES).issubset(set(results["species_counts"].keys()))
    results["validation_passed"] = (
        results["all_records_accounted"]
        and len(results["failed_records"]) == 0
        and results["all_species_represented"]
    )

    return results


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image preprocessing and validation module for WildMatch AI."
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset/curated/metadata.csv",
        help="Path to curated metadata CSV (default: dataset/curated/metadata.csv).",
    )
    parser.add_argument(
        "--save-samples-dir",
        type=str,
        default=None,
        help="Optional directory to save sample preprocessed images for visual inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting image preprocessing validation...")
    logger.info("Metadata CSV: %s", args.metadata_csv)

    results = validate_curated_images(
        metadata_csv=args.metadata_csv,
        save_samples_dir=args.save_samples_dir,
    )

    logger.info("--- Image Preprocessing Validation Results ---")
    logger.info("Total curated records: %d", results["total_records"])
    logger.info("Unique photo IDs: %d", results["unique_photos"])
    logger.info("Successfully processed images: %d / %d", results["processed_success"], results["total_records"])
    logger.info("Species distribution: %s", results["species_counts"])
    logger.info("Aspect ratio distribution: %s", results["aspect_ratios"])
    logger.info("Failed records: %d", len(results["failed_records"]))
    logger.info("Validation passed: %s", results["validation_passed"])

    if not results["validation_passed"]:
        logger.error("Validation failed! Details: %s", results["failed_records"])
        sys.exit(1)

    print("\n" + "=" * 80)
    print("REPRESENTATIVE IMAGE PREPROCESSING SAMPLES (ONE PER SPECIES)")
    print("=" * 80)
    for species, info in results["sample_records"].items():
        print(f"Species: {species} (Photo ID: {info['photo_id']})")
        print(f"  Source Path        : {info['image_path']}")
        print(f"  Original Size      : {info['orig_dimensions']} (Aspect Ratio: {info['aspect_ratio']})")
        print(f"  Resized Dimensions : {info['resized_dimensions']} ({info['padding_offsets']})")
        print(f"  Padded Square Size : 224x224 (RGB)")
        print(f"  PyTorch Tensor     : shape={info['tensor_shape']}, dtype={info['tensor_dtype']}")
        print(f"  Pixel Value Range  : {info['pixel_range']}")
        if "sample_file" in info:
            print(f"  Sample Saved At    : {info['sample_file']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
