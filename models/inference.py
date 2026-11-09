"""
WildMatch AI - Version 3 Inference API and Reusable Utilities.

Provides high-level, production-ready inference functions for Member 4
and downstream applications:
- load_model: loads WildMatchV3DualEncoder from checkpoint.
- load_species_heads: loads auxiliary classification heads from checkpoint.
- preprocess_image: loads, aspect-ratio resizes, pads (224x224), normalizes (ImageNet).
- tokenize_caption: cleans, tokenizes, and converts caption to integer token tensor [1, 32].
- embed_image: generates [B, 256] unit L2 image embedding.
- embed_text: generates [B, 256] unit L2 text embedding.
- compute_similarity: computes cosine similarity matrix between visual and textual embeddings.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn.functional as F
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageOps
import numpy as np

try:
    from models.wildmatch_v3 import (
        WildMatchV3DualEncoder,
        WildMatchV3SpeciesHeads,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SPECIES_CLASSES,
    )
except ImportError:
    from WildMatch_AI.models.wildmatch_v3 import (
        WildMatchV3DualEncoder,
        WildMatchV3SpeciesHeads,
        IMAGENET_MEAN,
        IMAGENET_STD,
        SPECIES_CLASSES,
    )


# ---------------------------------------------------------------------------
# Global Vocabulary Cache
# ---------------------------------------------------------------------------
_CACHED_VOCAB: Optional[Dict[str, Any]] = None
_CACHED_VOCAB_PATH: Optional[str] = None


def load_vocabulary(
    vocab_path: Union[str, Path] = "models/wildmatch_v3_vocabulary.json"
) -> Dict[str, Any]:
    """
    Load vocabulary dictionary from JSON file with in-memory caching.
    """
    global _CACHED_VOCAB, _CACHED_VOCAB_PATH

    path_obj = Path(vocab_path)
    if not path_obj.is_file():
        # Fallback to root or dataset/ directory
        candidates = [
            Path(path_obj.name),
            Path("models") / path_obj.name,
            Path("dataset/vocabulary.json"),
        ]
        for c in candidates:
            if c.is_file():
                path_obj = c
                break

    resolved = str(path_obj.resolve())
    if _CACHED_VOCAB is not None and _CACHED_VOCAB_PATH == resolved:
        return _CACHED_VOCAB

    with open(path_obj, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    _CACHED_VOCAB = vocab
    _CACHED_VOCAB_PATH = resolved
    return vocab


# ---------------------------------------------------------------------------
# Reusable Preprocessing Functions
# ---------------------------------------------------------------------------
def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    target_size: int = 224,
    apply_imagenet_norm: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Reusable image preprocessing function for Version 3 ResNet-18 dual encoder:
    1. Safe image loading with EXIF orientation correction and RGB conversion.
    2. Aspect-preserving resize with symmetric zero-padding (black letterbox) to target_size x target_size.
    3. Conversion to PyTorch float32 tensor [1, 3, target_size, target_size] in range [0.0, 1.0].
    4. Optional ImageNet mean/std normalization.

    Args:
        image_input: File path (str/Path) or existing PIL Image in any mode.
        target_size: Target square image dimension (default: 224).
        apply_imagenet_norm: Apply ImageNet mean/std normalization (default: True).
        device: Device to place the returned tensor on (default: 'cpu').

    Returns:
        PyTorch float32 tensor of shape [1, 3, target_size, target_size].
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_input}")
        with Image.open(img_path) as raw_img:
            img = ImageOps.exif_transpose(raw_img)
            if img is None:
                img = raw_img
            if img.mode != "RGB":
                img = img.convert("RGB")
            else:
                img = img.copy()
    elif isinstance(image_input, Image.Image):
        img = ImageOps.exif_transpose(image_input)
        if img is None:
            img = image_input
        if img.mode != "RGB":
            img = img.convert("RGB")
        else:
            img = img.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    # Aspect-preserving scale
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    # Convert to float32 tensor [0, 1]
    np_img = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)  # [3, 224, 224]

    # ImageNet normalization
    if apply_imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

    # Add batch dimension [1, 3, 224, 224]
    return tensor.unsqueeze(0).to(device)


def clean_caption_text(caption: str) -> str:
    """
    Normalize caption text:
    - Lowercase.
    - Remove punctuation (preserve alphanumeric and spaces).
    - Collapse extra whitespace.
    """
    if not caption or not isinstance(caption, str):
        return ""
    text = caption.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_caption(
    caption: str,
    vocab_or_path: Union[str, Path, Dict[str, Any]] = "models/wildmatch_v3_vocabulary.json",
    max_len: int = 32,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Reusable caption tokenization function for Version 3 BiLSTM text encoder:
    1. Cleans caption text (lowercase, strip punctuation, collapse whitespace).
    2. Splits into whitespace-delimited word tokens.
    3. Numericalizes using vocabulary mapping with BOS (<BOS> = 2) and EOS (<EOS> = 3).
    4. Truncates or pads with PAD (<PAD> = 0) to exactly max_len (32).

    Args:
        caption: Query or candidate description string.
        vocab_or_path: Vocabulary dictionary or path to wildmatch_v3_vocabulary.json.
        max_len: Fixed sequence length (default: 32).
        device: Device to place the returned tensor on (default: 'cpu').

    Returns:
        PyTorch long tensor of shape [1, max_len].
    """
    if isinstance(vocab_or_path, (str, Path)):
        vocab = load_vocabulary(vocab_or_path)
    elif isinstance(vocab_or_path, dict):
        vocab = vocab_or_path
    else:
        raise TypeError(f"Invalid vocab_or_path type: {type(vocab_or_path)}")

    word2idx = vocab.get("word2idx", vocab)
    pad_idx = word2idx.get("<PAD>", 0)
    unk_idx = word2idx.get("<UNK>", 1)
    bos_idx = word2idx.get("<BOS>", 2)
    eos_idx = word2idx.get("<EOS>", 3)

    cleaned = clean_caption_text(caption)
    words = cleaned.split() if cleaned else []

    token_ids = [word2idx.get(w, unk_idx) for w in words]
    token_ids = [bos_idx] + token_ids + [eos_idx]

    # Truncate
    if len(token_ids) > max_len:
        token_ids = token_ids[: max_len - 1] + [eos_idx]

    # Pad
    if len(token_ids) < max_len:
        token_ids = token_ids + [pad_idx] * (max_len - len(token_ids))

    return torch.tensor([token_ids], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Reusable Model Loading & Embedding Functions
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path: Union[str, Path] = "models/wildmatch_v3_dual_encoder.pth",
    device: Union[str, torch.device] = "cpu",
) -> WildMatchV3DualEncoder:
    """
    Load the trained WildMatch AI Version 3 Dual Encoder model.
    Instantiates ResNet-18 without downloading pretrained weights,
    then applies the trained state dictionary.

    Args:
        checkpoint_path: Path to wildmatch_v3_dual_encoder.pth.
        device: Torch device (default: 'cpu').

    Returns:
        WildMatchV3DualEncoder in eval mode on specified device.
    """
    model = WildMatchV3DualEncoder.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=str(device)
    )
    return model


def load_species_heads(
    checkpoint_path: Union[str, Path] = "models/wildmatch_v3_species_heads.pth",
    device: Union[str, torch.device] = "cpu",
) -> WildMatchV3SpeciesHeads:
    """
    Load auxiliary species classification heads.
    """
    return WildMatchV3SpeciesHeads.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=str(device)
    )


def embed_image(
    model: WildMatchV3DualEncoder,
    image_tensor: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Generate image embedding vector(s) using the trained dual encoder.

    Args:
        model: WildMatchV3DualEncoder instance.
        image_tensor: Image tensor [B, 3, 224, 224] or [3, 224, 224].
        normalize: Enforce unit L2 norm along dim=-1 (default: True).

    Returns:
        PyTorch float32 tensor of shape [B, 256].
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode_image(image_tensor, normalize=normalize)
    return embeddings


def embed_text(
    model: WildMatchV3DualEncoder,
    caption_input: Union[torch.Tensor, str, List[str]],
    vocab_or_path: Union[str, Path, Dict[str, Any]] = "models/wildmatch_v3_vocabulary.json",
    normalize: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Generate text embedding vector(s) using the trained dual encoder.
    Accepts pre-tokenized tensor [B, 32] or raw string/list of strings.

    Args:
        model: WildMatchV3DualEncoder instance.
        caption_input: Raw string, list of strings, or token ID tensor [B, 32].
        vocab_or_path: Path or dict of vocabulary.
        normalize: Enforce unit L2 norm along dim=-1 (default: True).
        device: Torch device.

    Returns:
        PyTorch float32 tensor of shape [B, 256].
    """
    if isinstance(caption_input, str):
        tokens = tokenize_caption(caption_input, vocab_or_path=vocab_or_path, device=device)
    elif isinstance(caption_input, list):
        tokens = torch.cat(
            [tokenize_caption(c, vocab_or_path=vocab_or_path, device=device) for c in caption_input],
            dim=0,
        )
    elif isinstance(caption_input, torch.Tensor):
        tokens = caption_input.to(device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported caption input type: {type(caption_input)}")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode_text(tokens, normalize=normalize)
    return embeddings


def compute_similarity(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between visual and textual embeddings.

    Args:
        image_embeddings: [N_img, 256] unit L2 normalized tensor.
        text_embeddings: [N_txt, 256] unit L2 normalized tensor.

    Returns:
        [N_img, N_txt] cosine similarity matrix.
    """
    return torch.matmul(image_embeddings, text_embeddings.t())
