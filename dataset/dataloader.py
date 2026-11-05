"""
WildMatch AI - DataLoader Setup Module

Provides factory functions to instantiate PyTorch DataLoaders for
train, validation, and test splits with standardized batching.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
from torch.utils.data import DataLoader

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from dataset.wildlife_dataset import WildlifeDataset
from preprocessing.text_preprocessing import Vocabulary

DEFAULT_BATCH_SIZE: int = 16
DEFAULT_NUM_WORKERS: int = 0  # Safe default for Windows compatibility


def create_dataloader(
    dataset: WildlifeDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a single DataLoader for a WildlifeDataset instance.

    Args:
        dataset: WildlifeDataset instance.
        batch_size: Number of image-caption pairs per batch (default: 16).
        shuffle: Whether to shuffle records every epoch (default: False).
        num_workers: Subprocesses for data loading (default: 0 for Windows).
        pin_memory: If True, copies Tensors to CUDA pinned memory before returning.
                    Defaults to True if CUDA is available, else False.
        drop_last: Set to True to drop the last incomplete batch.

    Returns:
        PyTorch DataLoader instance.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloaders(
    splits_dir: Union[str, Path] = "dataset/splits",
    vocab: Union[str, Path, Vocabulary] = "dataset/vocabulary.json",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: Optional[bool] = None,
    train_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eval_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    drop_last_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for all three splits (train, validation, test).

    Args:
        splits_dir: Directory containing train.csv, validation.csv, test.csv.
        vocab: Vocabulary instance or path to vocabulary.json.
        batch_size: Batch size for all loaders (default: 16).
        num_workers: Worker processes (default: 0).
        pin_memory: Whether to pin memory (default: auto-detected via torch.cuda.is_available()).
        train_transform: Optional image transform for training.
        eval_transform: Optional image transform for validation/testing.
        drop_last_train: Whether to drop the last incomplete training batch.

    Returns:
        tuple of (train_loader, val_loader, test_loader).
    """
    splits_path = Path(splits_dir)
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "validation.csv"
    test_csv = splits_path / "test.csv"

    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.is_file():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

    # Shared vocabulary loaded once if passed as path
    if isinstance(vocab, (str, Path)):
        vocab_obj = Vocabulary.load_json(str(vocab))
    else:
        vocab_obj = vocab

    # Instantiate datasets
    train_dataset = WildlifeDataset(
        split_csv=train_csv,
        vocab=vocab_obj,
        transform=train_transform,
    )
    val_dataset = WildlifeDataset(
        split_csv=val_csv,
        vocab=vocab_obj,
        transform=eval_transform,
    )
    test_dataset = WildlifeDataset(
        split_csv=test_csv,
        vocab=vocab_obj,
        transform=eval_transform,
    )

    # Instantiate loaders
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Training loader must shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Validation loader must not shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test loader must not shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# Backward-compatible alias
get_dataloaders = create_dataloaders
