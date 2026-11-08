# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ImageEncoder(nn.Module):
    """
    Custom Convolutional Neural Network (CNN) Image Encoder for WildMatch AI.

    Extracts hierarchical spatial visual features from standardized wildlife image
    tensors [B, 3, 224, 224] and projects them into a shared, L2-normalized
    embedding space [B, embed_dim] (default dimension 256).

    Architecture:
        - 4 Convolutional stages with BatchNorm2d, ReLU, and MaxPool2d:
            Block 1: Conv(in_channels -> 32, k=3, s=1, p=1) -> BN -> ReLU -> MaxPool(2x2)  => [B, 32, 112, 112]
            Block 2: Conv(32 -> 64, k=3, s=1, p=1)          -> BN -> ReLU -> MaxPool(2x2)  => [B, 64, 56, 56]
            Block 3: Conv(64 -> 128, k=3, s=1, p=1)         -> BN -> ReLU -> MaxPool(2x2)  => [B, 128, 28, 28]
            Block 4: Conv(128 -> feature_dim, k=3, s=1, p=1)-> BN -> ReLU -> MaxPool(2x2)  => [B, feature_dim, 14, 14]
        - Spatial Pooling: AdaptiveAvgPool2d((1, 1)) + Flatten => [B, feature_dim]
        - Projection Head: Two-layer MLP with BatchNorm1d, ReLU, Dropout, and Linear => [B, embed_dim]
        - Normalization: L2 unit normalization (norm = 1.0) along dim=-1 when normalize=True.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        feature_dim: int = 256,
        dropout_p: float = 0.1,
    ) -> None:
        """
        Initialize the ImageEncoder.

        Args:
            in_channels: Number of input image channels (default: 3 for RGB).
            embed_dim: Shared projection embedding dimension (default: 256, supports 128, 512).
            feature_dim: Feature channel dimension exiting the convolutional backbone (default: 256).
            dropout_p: Dropout probability in the projection head (default: 0.1).
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim
        self.feature_dim: int = feature_dim
        self.dropout_p: float = dropout_p

        # 4-stage convolutional backbone
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.backbone = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

        # Spatial aggregation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Projection head: 2-layer MLP mapping visual features to shared embedding space
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, embed_dim),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape and channel specifications.

        Args:
            x: Input image tensor.

        Raises:
            ValueError: If x is not a 4D tensor or channel count does not match in_channels.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor [batch_size, channels, height, width], "
                f"but received tensor with {x.ndim} dimension(s): shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s) at dim 1, "
                f"but received {x.shape[1]} channel(s): shape={tuple(x.shape)}"
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate spatial visual feature maps from the CNN backbone.

        Args:
            x: Standardized image tensor [B, in_channels, H, W] (e.g. [B, 3, 224, 224]).

        Returns:
            Spatial feature map tensor [B, feature_dim, H/16, W/16] (e.g. [B, 256, 14, 14]).
        """
        self._validate_input(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass mapping input images to shared embedding space.

        Args:
            x: Standardized image tensor [B, in_channels, H, W] (e.g. [B, 3, 224, 224]).
            normalize: If True, applies L2 unit normalization along embedding dimension.

        Returns:
            Image embeddings [B, embed_dim] (norm = 1.0 when normalize=True).
        """
        self._validate_input(x)
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = self.flatten(pooled)
        projected = self.projection_head(flat)

        if normalize:
            return F.normalize(projected, p=2, dim=-1, eps=1e-12)
        return projected

    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the output embedding.

        Returns:
            Embedding dimension integer (e.g. 256).
        """
        return self.embed_dim
