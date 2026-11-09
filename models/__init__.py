"""
WildMatch AI - Models Package

Exposes deep learning architectures and inference utilities for multimodal
wildlife image-text matching and species classification.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

try:
    from models.image_encoder import ImageEncoder
    from models.text_encoder import TextEncoder
    from models.dual_encoder import DualEncoder
    from models.wildmatch_v3 import (
        WildMatchV3DualEncoder,
        WildMatchV3ImageEncoder,
        WildMatchV3TextEncoder,
        WildMatchV3SpeciesHeads,
        SPECIES_CLASSES,
        SPECIES_TO_IDX,
        IDX_TO_SPECIES,
    )
    from models.inference import (
        load_model,
        load_species_heads,
        preprocess_image,
        tokenize_caption,
        embed_image,
        embed_text,
        compute_similarity,
    )
except ImportError:
    from WildMatch_AI.models.image_encoder import ImageEncoder
    from WildMatch_AI.models.text_encoder import TextEncoder
    from WildMatch_AI.models.dual_encoder import DualEncoder
    from WildMatch_AI.models.wildmatch_v3 import (
        WildMatchV3DualEncoder,
        WildMatchV3ImageEncoder,
        WildMatchV3TextEncoder,
        WildMatchV3SpeciesHeads,
        SPECIES_CLASSES,
        SPECIES_TO_IDX,
        IDX_TO_SPECIES,
    )
    from WildMatch_AI.models.inference import (
        load_model,
        load_species_heads,
        preprocess_image,
        tokenize_caption,
        embed_image,
        embed_text,
        compute_similarity,
    )

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "DualEncoder",
    "WildMatchV3DualEncoder",
    "WildMatchV3ImageEncoder",
    "WildMatchV3TextEncoder",
    "WildMatchV3SpeciesHeads",
    "SPECIES_CLASSES",
    "SPECIES_TO_IDX",
    "IDX_TO_SPECIES",
    "load_model",
    "load_species_heads",
    "preprocess_image",
    "tokenize_caption",
    "embed_image",
    "embed_text",
    "compute_similarity",
]
