# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Optional, Tuple

try:
    from WildMatch_AI.models.image_encoder import ImageEncoder
    from WildMatch_AI.models.text_encoder import TextEncoder
except ImportError:
    from models.image_encoder import ImageEncoder
    from models.text_encoder import TextEncoder


class DualEncoder(nn.Module):
    """
    Unified Dual-Encoder Multimodal Architecture for WildMatch AI.

    Composes the Custom CNN ImageEncoder and the BiLSTM TextEncoder to map wildlife
    images [B_img, 3, 224, 224] and tokenized captions [B_txt, 32] into a shared,
    calibrated, L2-normalized embedding space [B, embed_dim] (default dimension 256).

    Features:
        - Independent unimodal encoding endpoints (`encode_image`, `encode_text`)
        - Joint multimodal forward pass returning unit-normalized representations
        - Learnable temperature parameter (`logit_scale`) initialized to ln(1/0.07)
          and clamped to ln(max_temperature) for stable contrastive training
        - Cross-modal cosine similarity evaluation (`compute_similarity`) supporting
          arbitrary batch dimensions [B_img, B_txt]
        - Symmetric temperature-scaled logits (`get_logits`) ready for Member 3
          symmetric InfoNCE contrastive loss optimization
    """

    def __init__(
        self,
        image_encoder: Optional[ImageEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        embed_dim: int = 256,
        init_temperature: float = 0.07,
        max_temperature: float = 100.0,
        image_kwargs: Optional[Dict[str, Any]] = None,
        text_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the DualEncoder.

        Args:
            image_encoder: Pre-instantiated ImageEncoder module. If None, instantiates
                a new ImageEncoder with embed_dim and image_kwargs.
            text_encoder: Pre-instantiated TextEncoder module. If None, instantiates
                a new TextEncoder with embed_dim and text_kwargs.
            embed_dim: Shared projection embedding dimension (default: 256, supports 128, 512).
            init_temperature: Initial temperature scaling value tau (default: 0.07).
                logit_scale is initialized to ln(1 / init_temperature) ~ 2.6593.
            max_temperature: Maximum upper bound clamp for temperature scaling (default: 100.0).
            image_kwargs: Additional keyword arguments for ImageEncoder constructor if instantiated.
            text_kwargs: Additional keyword arguments for TextEncoder constructor if instantiated.
        """
        super().__init__()
        self.embed_dim: int = embed_dim
        self.max_temperature: float = max_temperature

        # Sub-encoder composition
        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            img_args = image_kwargs or {}
            self.image_encoder = ImageEncoder(embed_dim=embed_dim, **img_args)

        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            txt_args = text_kwargs or {}
            self.text_encoder = TextEncoder(embed_dim=embed_dim, **txt_args)

        # Validate embedding dimension alignment across sub-encoders
        if self.image_encoder.embed_dim != self.embed_dim:
            raise ValueError(
                f"ImageEncoder embed_dim ({self.image_encoder.embed_dim}) does not match "
                f"DualEncoder embed_dim ({self.embed_dim})"
            )
        if self.text_encoder.embed_dim != self.embed_dim:
            raise ValueError(
                f"TextEncoder embed_dim ({self.text_encoder.embed_dim}) does not match "
                f"DualEncoder embed_dim ({self.embed_dim})"
            )

        # Learnable temperature scaling parameter (ln(1 / tau))
        # Initialized to ln(1 / 0.07) ~ 2.65926 (scale factor ~ 14.2857)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / init_temperature)
        )

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode an image batch into the shared embedding space.

        Args:
            images: Wildlife image tensor [B_img, 3, 224, 224] (float32).
            normalize: If True, applies L2 unit normalization along embedding dimension.

        Returns:
            Image embedding tensor [B_img, embed_dim].
        """
        return self.image_encoder(images, normalize=normalize)

    def encode_text(self, captions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode a caption token ID batch into the shared embedding space.

        Args:
            captions: Caption token ID tensor [B_txt, seq_len] (torch.long).
            normalize: If True, applies L2 unit normalization along embedding dimension.

        Returns:
            Text embedding tensor [B_txt, embed_dim].
        """
        return self.text_encoder(captions, normalize=normalize)

    def get_scale(self) -> torch.Tensor:
        """
        Return the clamped temperature scaling factor (exp(logit_scale)).

        Clamped to math.log(self.max_temperature) to prevent numerical instability
        and gradient explosion during contrastive training.

        Returns:
            Scalar tensor representing current scale factor (1 / tau).
        """
        max_log_scale = math.log(self.max_temperature)
        return self.logit_scale.clamp(max=max_log_scale).exp()

    @property
    def temperature(self) -> float:
        """
        Return the current effective temperature value tau (1 / scale).

        Returns:
            Effective temperature float (e.g. ~0.07 at initialization).
        """
        return float(1.0 / self.get_scale().item())

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joint forward pass encoding paired image and text batches.

        Args:
            images: Image tensor [B, 3, 224, 224].
            captions: Token ID tensor [B, seq_len].
            normalize: Whether to apply L2 unit normalization (default: True).

        Returns:
            Tuple of:
                - image_embeds: [B, embed_dim]
                - text_embeds: [B, embed_dim]
                - scale: scalar tensor (exp(logit_scale))
        """
        image_embeds = self.encode_image(images, normalize=normalize)
        text_embeds = self.encode_text(captions, normalize=normalize)
        return image_embeds, text_embeds, self.get_scale()

    def compute_similarity(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix between image and text batches.

        Supports rectangular batches where B_img != B_txt (e.g. retrieval candidate scoring).

        Args:
            images: Image tensor [B_img, 3, 224, 224].
            captions: Token ID tensor [B_txt, seq_len].

        Returns:
            Cosine similarity matrix [B_img, B_txt] with values in [-1.0, 1.0].
        """
        image_embeds = self.encode_image(images, normalize=True)
        text_embeds = self.encode_text(captions, normalize=True)
        # Cosine similarity is dot product of L2 unit vectors: [B_img, D] @ [D, B_txt] -> [B_img, B_txt]
        similarity = torch.matmul(image_embeds, text_embeds.t())
        return similarity

    def get_logits(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric temperature-scaled logits for contrastive loss optimization.

        Computes logits_per_image = scale * (image_embeds @ text_embeds.T)
        and logits_per_text = logits_per_image.T.

        Args:
            images: Image tensor [B_img, 3, 224, 224].
            captions: Token ID tensor [B_txt, seq_len].

        Returns:
            Tuple of:
                - logits_per_image: [B_img, B_txt] float32 tensor
                - logits_per_text: [B_txt, B_img] float32 tensor
        """
        similarity = self.compute_similarity(images, captions)
        scale = self.get_scale()
        logits_per_image = scale * similarity
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the shared multimodal embedding space.

        Returns:
            Embedding dimension integer (e.g. 256).
        """
        return self.embed_dim
