"""
WildMatch AI - Wildlife PyTorch Dataset Module

Provides a reusable PyTorch Dataset for loading curated wildlife images
and tokenized captions for multimodal image-text retrieval.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
from torch.utils.data import Dataset

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from preprocessing.image_preprocessing import preprocess_image
from preprocessing.text_preprocessing import (
    Vocabulary,
    tokenize_caption,
    clean_caption,
    MAX_SEQ_LENGTH,
)

# Standardized species-to-index mapping for classification / evaluation
SPECIES_TO_IDX: Dict[str, int] = {
    "Tiger": 0,
    "Lion": 1,
    "Asian Elephant": 2,
    "Chital": 3,
}
IDX_TO_SPECIES: Dict[int, str] = {v: k for k, v in SPECIES_TO_IDX.items()}


class WildlifeDataset(Dataset):
    """
    PyTorch Dataset for WildMatch AI wildlife images and captions.

    Returns:
        dict with keys:
            - 'image': FloatTensor of shape [3, 224, 224], pixel values in [0.0, 1.0]
            - 'caption': LongTensor of shape [32], padded/truncated word token indices
            - 'species': str, species common name (e.g. 'Tiger')
            - 'species_idx': LongTensor scalar, integer class index in [0, 3]
            - 'photo_id': int, original photo ID from source dataset
            - 'image_id': str, unique image identifier (e.g. 'tiger_759553570')
            - 'image_path': str, file path to the curated image
    """

    def __init__(
        self,
        split_csv: Union[str, Path, pd.DataFrame],
        vocab: Union[str, Path, Vocabulary] = "dataset/vocabulary.json",
        image_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_size: int = 224,
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        """
        Initialize the WildlifeDataset.

        Args:
            split_csv: File path to split CSV (train.csv, validation.csv, test.csv)
                       OR an existing pandas DataFrame.
            vocab: Vocabulary instance OR path to serialized vocabulary.json.
            image_root: Optional root directory override for image paths.
            transform: Optional callable transform to apply to the [3, 224, 224] image tensor.
            target_size: Target square image dimension (default: 224).
            max_seq_length: Target token sequence length including BOS and EOS (default: 32).
        """
        super().__init__()

        # Load metadata dataframe
        if isinstance(split_csv, pd.DataFrame):
            self.df = split_csv.copy().reset_index(drop=True)
            self.csv_path = None
        else:
            self.csv_path = Path(split_csv)
            if not self.csv_path.is_file():
                raise FileNotFoundError(f"Split CSV not found: {split_csv}")
            self.df = pd.read_csv(self.csv_path).reset_index(drop=True)

        # Load vocabulary
        if isinstance(vocab, Vocabulary):
            self.vocab = vocab
        else:
            vocab_path = Path(vocab)
            if not vocab_path.is_file():
                raise FileNotFoundError(f"Vocabulary JSON not found: {vocab}")
            self.vocab = Vocabulary.load_json(str(vocab_path))

        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.target_size = target_size
        self.max_seq_length = max_seq_length

        self._validate_schema()

    def _validate_schema(self) -> None:
        """Validate required columns exist in the DataFrame."""
        required_cols = {"image_id", "photo_id", "species", "image_path"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Dataset DataFrame missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> str:
        """Resolve image path with optional image_root override."""
        path_obj = Path(raw_path)
        if self.image_root is not None:
            # Rebase under image_root if needed
            return str(self.image_root / path_obj.name)
        return str(path_obj)

    def _get_tokens(self, row: pd.Series) -> List[str]:
        """Extract or tokenize words for the row."""
        if "tokens" in row and pd.notna(row["tokens"]):
            val = row["tokens"]
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    return val.split()
            elif isinstance(val, (list, tuple)):
                return list(val)

        # Fallback to tokenizing cleaned_caption or final_caption
        caption_text = row.get("cleaned_caption") or row.get("final_caption") or ""
        return tokenize_caption(caption_text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetch a single image-caption pair by index.

        Returns:
            dict containing image tensor, token indices tensor, and metadata.
        """
        row = self.df.iloc[idx]

        # 1. Image preprocessing
        image_path = self._resolve_image_path(str(row["image_path"]))
        image_tensor = preprocess_image(image_path, target_size=self.target_size)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # 2. Text tokenization & numericalization
        tokens = self._get_tokens(row)
        token_ids = self.vocab.numericalize(
            tokens,
            add_special_tokens=True,
            max_len=self.max_seq_length,
            pad=True,
            truncate=True,
        )
        caption_tensor = torch.tensor(token_ids, dtype=torch.long)

        # 3. Species indexing
        species_str = str(row["species"])
        species_idx = SPECIES_TO_IDX.get(species_str, -1)
        species_idx_tensor = torch.tensor(species_idx, dtype=torch.long)

        # 4. IDs
        photo_id = int(row["photo_id"])
        image_id = str(row["image_id"])

        return {
            "image": image_tensor,
            "caption": caption_tensor,
            "species": species_str,
            "species_idx": species_idx_tensor,
            "photo_id": photo_id,
            "image_id": image_id,
            "image_path": image_path,
        }

    def decode_caption(self, caption_tensor: torch.Tensor, remove_special_tokens: bool = True) -> str:
        """Utility method to decode a token index tensor back to a readable string."""
        indices = caption_tensor.tolist()
        words = self.vocab.decode(indices, remove_special_tokens=remove_special_tokens)
        return " ".join(words)
