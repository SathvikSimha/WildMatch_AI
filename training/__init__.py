"""
WildMatch AI - Training & Loss Package (Member 3)

Provides modular contrastive loss functions, auxiliary classification objectives,
and end-to-end training loops for dual-encoder vision-language models.
"""

from training.loss import (
    SymmetricContrastiveLoss,
    SpeciesAwareMultiPositiveContrastiveLoss,
    AuxiliaryClassificationLoss,
    WildMatchLoss,
)

__all__ = [
    "SymmetricContrastiveLoss",
    "SpeciesAwareMultiPositiveContrastiveLoss",
    "AuxiliaryClassificationLoss",
    "WildMatchLoss",
]

