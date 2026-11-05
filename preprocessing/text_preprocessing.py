"""
WildMatch AI - Text Preprocessing Module

This module implements reproducible text preprocessing for image captions:
1. Text cleaning (lowercase conversion, punctuation stripping, whitespace normalization, safe null handling).
2. Word-level tokenization for caption sentences.
3. Token sequence padding and truncation support (max length 32 tokens).
4. Custom Vocabulary definition for the BiLSTM text encoder (special tokens: <PAD>, <UNK>, <BOS>, <EOS>).
   NOTE: In strict compliance with data-leakage prevention rules, vocabulary construction
   is deferred until after the train/validation/test split and MUST only be fitted on training captions.
5. Export to `dataset/captions/processed_captions.csv` preserving original columns and all 200 records.
"""

# Import torch first if available to avoid Windows OpenMP/MKL DLL initialization conflicts
try:
    import torch
except (ImportError, OSError):
    pass

import argparse
from collections import Counter
import json
import logging
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_TextPreprocessing")

# ---------------------------------------------------------------------------
# Tokenization and Sequence Constants
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

MAX_SEQ_LENGTH = 32
MIN_VOCAB_FREQ = 1

EXPECTED_RECORD_COUNT = 200
EXPECTED_PER_SPECIES = 50
EXPECTED_SPECIES = {"Tiger", "Lion", "Asian Elephant", "Chital"}

# ---------------------------------------------------------------------------
# Text Cleaning & Word Tokenization Functions
# ---------------------------------------------------------------------------
def clean_caption(caption: Any) -> str:
    """
    Clean caption text safely:
    - Handles empty, null, or non-string inputs gracefully.
    - Converts text to lowercase.
    - Strips unnecessary punctuation while preserving alphanumeric words.
    - Normalizes whitespace (collapsing repeated spaces, tabs, and newlines).

    Args:
        caption: Input caption (string or pandas NaN/None).

    Returns:
        Cleaned lowercase string without punctuation or irregular whitespace.
    """
    if caption is None or pd.isna(caption):
        return ""

    text = str(caption).strip()
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary punctuation (keep alphanumeric characters and whitespace)
    # Replaces any character that is not a word character (a-z, 0-9, _) or whitespace
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace: collapse multiple spaces/tabs/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_caption(caption: Any) -> List[str]:
    """
    Perform word-level tokenization on a caption:
    - Cleans the caption first using `clean_caption()`.
    - Splits cleaned text into individual word tokens based on whitespace.
    - Returns an empty list if caption is empty or null.

    Args:
        caption: Input caption text.

    Returns:
        List of lowercase word token strings.
    """
    cleaned = clean_caption(caption)
    if not cleaned:
        return []
    return cleaned.split()


def pad_and_truncate_tokens(
    tokens: Sequence[str],
    max_len: int = MAX_SEQ_LENGTH,
    pad_token: str = PAD_TOKEN,
    add_special_tokens: bool = False,
    bos_token: str = BOS_TOKEN,
    eos_token: str = EOS_TOKEN,
) -> List[str]:
    """
    Pad or truncate a list of string tokens to a fixed sequence length.

    Args:
        tokens: Sequence of word token strings.
        max_len: Fixed maximum sequence length (default: 32).
        pad_token: Token used for right-padding (default: "<PAD>").
        add_special_tokens: Whether to prepend BOS and append EOS tokens.
        bos_token: Beginning of sequence token.
        eos_token: End of sequence token.

    Returns:
        Padded and/or truncated list of tokens with exact length `max_len`.
    """
    seq = list(tokens)
    if add_special_tokens:
        seq = [bos_token] + seq + [eos_token]

    if len(seq) > max_len:
        # Truncate sequence
        if add_special_tokens:
            seq = seq[: max_len - 1] + [eos_token]
        else:
            seq = seq[:max_len]
    elif len(seq) < max_len:
        # Pad sequence with pad_token
        seq = seq + [pad_token] * (max_len - len(seq))

    return seq


# ---------------------------------------------------------------------------
# Custom Vocabulary Definition for BiLSTM
# ---------------------------------------------------------------------------
class Vocabulary:
    """
    Custom word-level Vocabulary for the BiLSTM text encoder.
    Independent of CLIP's pretrained tokenizer.

    IMPORTANT:
    To prevent data leakage, vocabulary construction must be executed
    ONLY on the training split, never on the full dataset or validation/test sets.
    """

    def __init__(
        self,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        max_seq_length: int = MAX_SEQ_LENGTH,
        min_freq: int = MIN_VOCAB_FREQ,
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_seq_length = max_seq_length
        self.min_freq = min_freq

        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freqs: Counter = Counter()
        self._is_built: bool = False

        # Initialize special tokens with fixed indexes
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self.word2idx[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.eos_token]

    @property
    def is_built(self) -> bool:
        return self._is_built

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, self.unk_idx)

    def build_vocab(self, tokenized_captions: Sequence[Sequence[str]]) -> None:
        """
        Build vocabulary exclusively from training set tokenized captions.
        Filters out tokens that appear less than `min_freq` times.

        Args:
            tokenized_captions: Iterable of token lists from training data.
        """
        self.word_freqs = Counter()
        for tokens in tokenized_captions:
            self.word_freqs.update(tokens)

        current_idx = len(self.special_tokens)
        for word, freq in self.word_freqs.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1

        self._is_built = True
        logger.info(
            "Vocabulary constructed with %d total tokens (%d special tokens, %d words).",
            len(self.word2idx),
            len(self.special_tokens),
            len(self.word2idx) - len(self.special_tokens),
        )

    def numericalize(
        self,
        tokens: Sequence[str],
        add_special_tokens: bool = True,
        max_len: Optional[int] = None,
        pad: bool = True,
        truncate: bool = True,
    ) -> List[int]:
        """
        Convert token strings to vocabulary integer IDs.

        Args:
            tokens: Sequence of word tokens.
            add_special_tokens: Whether to add BOS and EOS tokens.
            max_len: Target sequence length (defaults to `self.max_seq_length`).
            pad: Whether to pad with PAD token to `max_len`.
            truncate: Whether to truncate sequences exceeding `max_len`.

        Returns:
            List of integer token IDs.
        """
        if not self._is_built:
            raise RuntimeError(
                "Vocabulary has not been built yet. Build vocabulary using training split captions "
                "before numericalizing."
            )

        target_len = max_len or self.max_seq_length
        token_ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]

        if add_special_tokens:
            token_ids = [self.bos_idx] + token_ids + [self.eos_idx]

        if truncate and len(token_ids) > target_len:
            if add_special_tokens:
                token_ids = token_ids[: target_len - 1] + [self.eos_idx]
            else:
                token_ids = token_ids[:target_len]

        if pad and len(token_ids) < target_len:
            token_ids = token_ids + [self.pad_idx] * (target_len - len(token_ids))

        return token_ids

    def decode(
        self, indices: Sequence[int], remove_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert integer IDs back to string tokens.
        """
        tokens = []
        for idx in indices:
            token = self.idx2word.get(idx, self.unk_token)
            if remove_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary state to a dictionary."""
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "max_seq_length": self.max_seq_length,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_freqs": dict(self.word_freqs),
            "is_built": self._is_built,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        """Reconstruct vocabulary instance from serialized dictionary."""
        vocab = cls(
            pad_token=data.get("pad_token", PAD_TOKEN),
            unk_token=data.get("unk_token", UNK_TOKEN),
            bos_token=data.get("bos_token", BOS_TOKEN),
            eos_token=data.get("eos_token", EOS_TOKEN),
            max_seq_length=data.get("max_seq_length", MAX_SEQ_LENGTH),
            min_freq=data.get("min_freq", MIN_VOCAB_FREQ),
        )
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_freqs = Counter(data.get("word_freqs", {}))
        vocab._is_built = data.get("is_built", True)
        return vocab

    def save_json(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved vocabulary to %s", filepath)

    @classmethod
    def load_json(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Preprocessing Pipeline Functions
# ---------------------------------------------------------------------------
def preprocess_captions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process caption dataframe:
    - Preserves all original columns and rows.
    - Cleans `final_caption` into `cleaned_caption`.
    - Tokenizes `cleaned_caption` into `tokens` (serialized as JSON list string).

    Args:
        df: Input DataFrame containing `final_caption`.

    Returns:
        New DataFrame with `cleaned_caption` and `tokens` columns added.
    """
    if "final_caption" not in df.columns:
        raise KeyError("Input dataframe must contain a 'final_caption' column.")

    processed_df = df.copy()

    # Generate cleaned caption
    processed_df["cleaned_caption"] = processed_df["final_caption"].apply(clean_caption)

    # Generate token lists and serialize as JSON list string for robust CSV persistence
    token_lists = processed_df["cleaned_caption"].apply(lambda text: text.split() if text else [])
    processed_df["tokens"] = token_lists.apply(json.dumps)

    # Reorder columns to place requested columns first, followed by any remaining original columns
    preferred_order = [
        "image_id",
        "photo_id",
        "species",
        "image_path",
        "blip_caption",
        "final_caption",
        "cleaned_caption",
        "tokens",
    ]
    existing_cols = [c for c in preferred_order if c in processed_df.columns]
    extra_cols = [c for c in processed_df.columns if c not in existing_cols]
    final_cols = existing_cols + extra_cols

    return processed_df[final_cols]


def validate_preprocessed_data(
    processed_df: pd.DataFrame,
    original_df: pd.DataFrame,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
    expected_per_species: int = EXPECTED_PER_SPECIES,
) -> Dict[str, Any]:
    """
    Validate preprocessed captions according to project requirements:
    1. Exactly expected unique photo IDs (default: 200).
    2. No missing final captions.
    3. No empty cleaned captions.
    4. Tokenization works and produces tokens for every record.
    5. Original captions were preserved and not modified.
    6. All four target species are represented and balanced (default: 50 each).
    7. Maximum sequence length is respected (<= 32 tokens).

    Returns:
        Dictionary of check results and boolean `all_passed`.
    """
    checks: Dict[str, Any] = {}

    # Check 1: Record count and unique photo IDs
    total_records = len(processed_df)
    unique_photos = processed_df["photo_id"].nunique()
    checks["record_count"] = total_records
    checks["unique_photo_ids"] = unique_photos
    checks["exact_expected_records"] = (
        total_records == expected_record_count and unique_photos == expected_record_count
    )

    # Check 2: No missing final captions
    missing_final = processed_df["final_caption"].isna().sum() + (processed_df["final_caption"].astype(str).str.strip() == "").sum()
    checks["missing_final_captions"] = int(missing_final)
    checks["no_missing_final"] = missing_final == 0

    # Check 3: No empty cleaned captions
    empty_cleaned = (processed_df["cleaned_caption"].str.strip() == "").sum()
    checks["empty_cleaned_captions"] = int(empty_cleaned)
    checks["no_empty_cleaned"] = empty_cleaned == 0

    # Check 4: Tokenization works for every record
    tokens_parsed = processed_df["tokens"].apply(json.loads)
    token_lengths = tokens_parsed.apply(len)
    empty_token_lists = (token_lengths == 0).sum()
    checks["min_token_length"] = int(token_lengths.min())
    checks["max_token_length"] = int(token_lengths.max())
    checks["avg_token_length"] = float(round(token_lengths.mean(), 2))
    checks["empty_token_records"] = int(empty_token_lists)
    checks["all_records_tokenized"] = empty_token_lists == 0

    # Check 5: No original captions were modified
    final_captions_intact = processed_df["final_caption"].equals(original_df["final_caption"])
    blip_captions_intact = processed_df["blip_caption"].equals(original_df["blip_caption"])
    checks["final_captions_unmodified"] = final_captions_intact
    checks["blip_captions_unmodified"] = blip_captions_intact
    checks["original_captions_preserved"] = final_captions_intact and blip_captions_intact

    # Check 6: All four species represented and balanced
    species_present = set(processed_df["species"].unique())
    species_counts = processed_df["species"].value_counts()
    checks["species_distribution"] = species_counts.to_dict()
    checks["all_species_represented"] = EXPECTED_SPECIES.issubset(species_present)
    checks["species_balanced"] = all(
        species_counts.get(s, 0) == expected_per_species for s in EXPECTED_SPECIES
    )

    # Check 7: Maximum sequence length respected
    checks["max_seq_length_respected"] = bool(int(token_lengths.max()) <= MAX_SEQ_LENGTH)

    all_passed = (
        checks["exact_expected_records"]
        and checks["no_missing_final"]
        and checks["no_empty_cleaned"]
        and checks["all_records_tokenized"]
        and checks["original_captions_preserved"]
        and checks["all_species_represented"]
        and checks["species_balanced"]
        and checks["max_seq_length_respected"]
    )
    checks["all_passed"] = all_passed
    return checks


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text preprocessing and tokenization preparation for WildMatch AI."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="dataset/captions/pilot_captions.csv",
        help="Path to input caption CSV (default: dataset/captions/pilot_captions.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="dataset/captions/processed_captions.csv",
        help="Path to save processed captions CSV (default: dataset/captions/processed_captions.csv).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_RECORD_COUNT,
        help=f"Expected total record count (default: {EXPECTED_RECORD_COUNT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing and validation without writing output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting text preprocessing pipeline...")
    logger.info("Input CSV: %s", args.input_csv)
    logger.info("Output CSV: %s", args.output_csv)

    if not os.path.exists(args.input_csv):
        logger.error("Input CSV not found at: %s", args.input_csv)
        sys.exit(1)

    original_df = pd.read_csv(args.input_csv)
    logger.info("Loaded %d rows from %s.", len(original_df), args.input_csv)

    processed_df = preprocess_captions_dataframe(original_df)
    validation = validate_preprocessed_data(
        processed_df, original_df, expected_record_count=args.expected_count
    )

    logger.info("--- Validation Results ---")
    logger.info("Total records: %d (Unique photo IDs: %d)", validation["record_count"], validation["unique_photo_ids"])
    logger.info("Missing final captions: %d", validation["missing_final_captions"])
    logger.info("Empty cleaned captions: %d", validation["empty_cleaned_captions"])
    logger.info("Token length stats: min=%d, max=%d, avg=%.2f", validation["min_token_length"], validation["max_token_length"], validation["avg_token_length"])
    logger.info("Original captions preserved: %s", validation["original_captions_preserved"])
    logger.info("Species distribution: %s", validation["species_distribution"])
    logger.info("Species balanced (50 each): %s", validation["species_balanced"])
    logger.info("Max sequence length respected (<= 32): %s", validation["max_seq_length_respected"])
    logger.info("All validation criteria passed: %s", validation["all_passed"])

    if not validation["all_passed"]:
        logger.error("Validation failed! Halting before output creation.")
        sys.exit(1)

    if not args.dry_run:
        # Create timestamped backup if output file already exists
        if os.path.exists(args.output_csv):
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{args.output_csv}.bak_{ts}"
            shutil.copy2(args.output_csv, backup_path)
            logger.info("Created timestamped backup of existing processed CSV at %s", backup_path)

        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        processed_df.to_csv(args.output_csv, index=False)
        logger.info("Successfully wrote %d processed records to %s.", len(processed_df), args.output_csv)
    else:
        logger.info("[Dry Run] Validation succeeded. File write skipped.")

    # Display 5 before/after examples across different species
    print("\n" + "=" * 80)
    print("FIVE BEFORE / AFTER TEXT PREPROCESSING EXAMPLES")
    print("=" * 80)
    sample_indices = [0, 1, 2, 3, 4]
    for idx in sample_indices:
        row = processed_df.iloc[idx]
        print(f"Record {idx + 1} [{row['species']} - Photo ID: {row['photo_id']}]:")
        print(f"  Before (final_caption)  : \"{row['final_caption']}\"")
        print(f"  After  (cleaned_caption): \"{row['cleaned_caption']}\"")
        print(f"  Tokens (tokens)         : {row['tokens']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
