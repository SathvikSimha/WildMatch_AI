"""
WildMatch AI - Models Package

Exposes deep learning architectures for multimodal wildlife image-text matching.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

try:
    from WildMatch_AI.models.image_encoder import ImageEncoder
    from WildMatch_AI.models.text_encoder import TextEncoder
    from WildMatch_AI.models.dual_encoder import DualEncoder
except ImportError:
    from models.image_encoder import ImageEncoder
    from models.text_encoder import TextEncoder
    from models.dual_encoder import DualEncoder

__all__ = ["ImageEncoder", "TextEncoder", "DualEncoder"]
