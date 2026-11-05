"""
WildMatch AI - Comprehensive DataLoader & Dataset Validation Script

Validates the PyTorch Dataset and DataLoader implementation against 15 strict project criteria:
1. Dataset lengths: train 140, validation 30, test 30.
2. Image batch shape [B, 3, 224, 224].
3. Caption batch shape [B, 32].
4. Image dtype float32.
5. Caption dtype torch.long.
6. Image pixel values within [0.0, 1.0].
7. All image tensors have 3 channels.
8. All caption indices within vocabulary range [0, len(vocab)-1].
9. Padding uses index 0 (<PAD>).
10. Unknown words map to index 1 (<UNK>).
11. Species labels are valid and consistent with integer class indices.
12. Photo IDs and image IDs are preserved.
13. No missing or corrupted image files across all 200 items.
14. Validation and test loaders do not shuffle.
15. Training loader shuffles.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
from torch.utils.data import RandomSampler, SequentialSampler

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from dataset.wildlife_dataset import WildlifeDataset, SPECIES_TO_IDX, IDX_TO_SPECIES
from dataset.dataloader import create_dataloaders
from preprocessing.text_preprocessing import Vocabulary


def main():
    print("=" * 80)
    print("WILDMATCH AI - PYTORCH DATASET & DATALOADER VALIDATION SUITE")
    print("=" * 80)

    # Load Vocabulary
    vocab_path = "dataset/vocabulary.json"
    assert os.path.exists(vocab_path), f"Vocabulary missing: {vocab_path}"
    vocab = Vocabulary.load_json(vocab_path)
    vocab_size = len(vocab)
    print(f"\n[INFO] Loaded vocabulary: {vocab_size} tokens (<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3)")

    # Instantiate DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        splits_dir="dataset/splits",
        vocab=vocab,
        batch_size=16,
        num_workers=0,
    )

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset

    # -------------------------------------------------------------------------
    # Check 1: Dataset lengths
    # -------------------------------------------------------------------------
    print("\n--- Check 1: Dataset Lengths ---")
    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)
    print(f"Train dataset length: {len_train} (expected 140)")
    print(f"Validation dataset length: {len_val} (expected 30)")
    print(f"Test dataset length: {len_test} (expected 30)")
    assert len_train == 140, f"Expected 140, got {len_train}"
    assert len_val == 30, f"Expected 30, got {len_val}"
    assert len_test == 30, f"Expected 30, got {len_test}"
    print("[PASS] Check 1: Dataset lengths are exactly 140, 30, 30.")

    # -------------------------------------------------------------------------
    # Check 13: Full image file loading integrity across all 200 records
    # -------------------------------------------------------------------------
    print("\n--- Check 13: Image File Loading & Integrity across All 200 Records ---")
    total_loaded = 0
    for split_name, ds in [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
        for idx in range(len(ds)):
            item = ds[idx]
            assert item["image"].shape == (3, 224, 224), f"Bad image shape at {split_name}[{idx}]"
            assert not torch.isnan(item["image"]).any(), f"NaN in image at {split_name}[{idx}]"
            total_loaded += 1
    print(f"Successfully loaded and verified all {total_loaded} images across train, validation, and test splits.")
    print("[PASS] Check 13: No missing or corrupted image files.")

    # -------------------------------------------------------------------------
    # Checks 2-12: Batch shape, dtype, pixel range, and token properties
    # -------------------------------------------------------------------------
    print("\n--- Checks 2-12: Batch Tensor Shapes, Dtypes & Content Validation ---")
    for split_name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        for batch_idx, batch in enumerate(loader):
            images = batch["image"]
            captions = batch["caption"]
            species = batch["species"]
            species_indices = batch["species_idx"]
            photo_ids = batch["photo_id"]
            image_ids = batch["image_id"]

            b_size = images.shape[0]

            # Check 2: Image batch shape [B, 3, 224, 224]
            assert images.ndim == 4, f"Image batch must have 4 dimensions, got {images.ndim}"
            assert images.shape == (b_size, 3, 224, 224), f"Unexpected image shape: {images.shape}"

            # Check 3: Caption batch shape [B, 32]
            assert captions.ndim == 2, f"Caption batch must have 2 dimensions, got {captions.ndim}"
            assert captions.shape == (b_size, 32), f"Unexpected caption shape: {captions.shape}"

            # Check 4: Image dtype float32
            assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"

            # Check 5: Caption dtype torch.long
            assert captions.dtype == torch.long, f"Expected torch.long, got {captions.dtype}"

            # Check 6: Image pixel values within [0.0, 1.0]
            min_val = images.min().item()
            max_val = images.max().item()
            assert min_val >= 0.0, f"Pixel value < 0: {min_val}"
            assert max_val <= 1.0, f"Pixel value > 1: {max_val}"

            # Check 7: 3 channels
            assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"

            # Check 8: Caption indices within vocabulary range [0, len(vocab)-1]
            assert (captions >= 0).all(), "Caption index < 0 found"
            assert (captions < vocab_size).all(), f"Caption index >= {vocab_size} found"

            # Check 9: Padding uses index 0 (<PAD>)
            # Every caption starts with <BOS>=2, has word tokens, ends with <EOS>=3, then trailing 0s
            for i in range(b_size):
                seq = captions[i].tolist()
                assert seq[0] == vocab.bos_idx == 2, f"Caption does not start with BOS: {seq}"
                assert vocab.eos_idx == 3 in seq, f"Caption missing EOS: {seq}"
                eos_pos = seq.index(vocab.eos_idx)
                for pad_idx in range(eos_pos + 1, 32):
                    assert seq[pad_idx] == vocab.pad_idx == 0, f"Padding position not 0: {seq[pad_idx]}"

            # Check 11: Species labels are valid and consistent
            for i in range(b_size):
                sp_name = species[i]
                sp_idx = species_indices[i].item()
                assert sp_name in SPECIES_TO_IDX, f"Invalid species name: {sp_name}"
                assert SPECIES_TO_IDX[sp_name] == sp_idx, f"Inconsistent species idx: {sp_name} vs {sp_idx}"
                assert IDX_TO_SPECIES[sp_idx] == sp_name, f"Reverse mapping failure for {sp_idx}"

            # Check 12: Photo IDs and Image IDs preserved
            for i in range(b_size):
                pid = photo_ids[i].item()
                img_id = image_ids[i]
                assert str(pid) in img_id, f"Photo ID {pid} not in image_id {img_id}"

    print("[PASS] Check 2: Image batch shape is [B, 3, 224, 224].")
    print("[PASS] Check 3: Caption batch shape is [B, 32].")
    print("[PASS] Check 4: Image tensor dtype is float32.")
    print("[PASS] Check 5: Caption tensor dtype is torch.long.")
    print(f"[PASS] Check 6: Image pixel values strictly in [0.0, 1.0] (observed min={min_val:.3f}, max={max_val:.3f}).")
    print("[PASS] Check 7: All image tensors have exactly 3 channels (RGB).")
    print(f"[PASS] Check 8: All caption indices strictly in range [0, {vocab_size - 1}].")
    print("[PASS] Check 9: Sequence padding strictly uses index 0 (<PAD>) following <EOS>.")
    print("[PASS] Check 11: Species labels and integer indices are 100% valid and consistent.")
    print("[PASS] Check 12: Photo IDs and Image IDs are correctly paired and preserved.")

    # -------------------------------------------------------------------------
    # Check 10: Unknown words map to index 1 (<UNK>)
    # -------------------------------------------------------------------------
    print("\n--- Check 10: UNK Token Mapping ---")
    test_oov_tokens = ["nonexistent_wildlife_token_12345", "xylophone_tiger_sample"]
    num_ids = vocab.numericalize(test_oov_tokens, add_special_tokens=True, max_len=8, pad=True)
    assert num_ids[1] == vocab.unk_idx == 1, f"Expected UNK index 1, got {num_ids[1]}"
    assert num_ids[2] == vocab.unk_idx == 1, f"Expected UNK index 1, got {num_ids[2]}"
    print(f"Sample OOV tokens '{test_oov_tokens}' correctly numericalized to: {num_ids}")
    print("[PASS] Check 10: Unknown words map strictly to index 1 (<UNK>).")

    # -------------------------------------------------------------------------
    # Check 14 & 15: Shuffling behavior
    # -------------------------------------------------------------------------
    print("\n--- Checks 14 & 15: DataLoader Shuffling Verification ---")
    # Validation sampler check
    assert isinstance(val_loader.sampler, SequentialSampler), "val_loader sampler is not SequentialSampler"
    val_pids_from_loader = []
    for batch in val_loader:
        val_pids_from_loader.extend(batch["photo_id"].tolist())
    val_pids_from_df = val_ds.df["photo_id"].tolist()
    assert val_pids_from_loader == val_pids_from_df, "Validation loader order does not match dataset order!"
    print("[PASS] Check 14a: Validation loader does NOT shuffle (matches sequential dataset order exactly).")

    # Test sampler check
    assert isinstance(test_loader.sampler, SequentialSampler), "test_loader sampler is not SequentialSampler"
    test_pids_from_loader = []
    for batch in test_loader:
        test_pids_from_loader.extend(batch["photo_id"].tolist())
    test_pids_from_df = test_ds.df["photo_id"].tolist()
    assert test_pids_from_loader == test_pids_from_df, "Test loader order does not match dataset order!"
    print("[PASS] Check 14b: Test loader does NOT shuffle (matches sequential dataset order exactly).")

    # Training sampler check
    assert isinstance(train_loader.sampler, RandomSampler), "train_loader sampler is not RandomSampler"
    train_pids_from_loader = []
    for batch in train_loader:
        train_pids_from_loader.extend(batch["photo_id"].tolist())
    train_pids_from_df = train_ds.df["photo_id"].tolist()

    # The order of photo IDs from shuffled train loader must differ from sequential order
    assert train_pids_from_loader != train_pids_from_df, "Train loader did not shuffle records!"
    assert sorted(train_pids_from_loader) == sorted(train_pids_from_df), "Train loader missing records!"
    print("[PASS] Check 15: Training loader DOES shuffle records (order differs from dataset order while preserving all 140 elements).")

    print("\n" + "=" * 80)
    print("ALL 15 DATALOADER & DATASET VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
