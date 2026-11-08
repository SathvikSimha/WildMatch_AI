# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TextEncoder(nn.Module):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) Text Encoder for WildMatch AI.

    Encodes standardized [B, 32] integer token ID sequences into a shared,
    L2-normalized multimodal embedding space [B, embed_dim] (default dimension 256).

    Architecture:
        - Word Embedding: Dense word embedding layer with dedicated padding index zeroing
        - Embedding Dropout: Regularization on word representations
        - Recurrent Backbone: 1-layer (configurable) Bidirectional LSTM extracting contextual representations
        - Sequence Pooling: Padding-aware temporal aggregation (masked mean, max, or last token)
        - Projection Head: Two-layer MLP with BatchNorm1d, ReLU, Dropout, and Linear
        - Normalization: L2 unit normalization (norm = 1.0) along dim=-1 when normalize=True
    """

    def __init__(
        self,
        vocab_size: int = 158,
        word_embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        embed_dim: int = 256,
        padding_idx: int = 0,
        dropout_p: float = 0.1,
        max_seq_length: int = 32,
        pooling: str = "mean",
    ) -> None:
        """
        Initialize the TextEncoder.

        Args:
            vocab_size: Size of the vocabulary (default: 158 from Member 1 training split).
            word_embed_dim: Dense word embedding dimension (default: 128).
            hidden_dim: BiLSTM hidden state dimension per direction (default: 128, producing 256 channels).
            num_layers: Number of stacked recurrent layers (default: 1).
            embed_dim: Shared projection embedding dimension (default: 256, matching ImageEncoder).
            padding_idx: Token index representing <PAD> (default: 0).
            dropout_p: Dropout probability applied to embeddings and projection head (default: 0.1).
            max_seq_length: Maximum allowed sequence length (default: 32).
            pooling: Temporal pooling strategy ('mean', 'max', or 'last').
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        self.word_embed_dim: int = word_embed_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.embed_dim: int = embed_dim
        self.padding_idx: int = padding_idx
        self.dropout_p: float = dropout_p
        self.max_seq_length: int = max_seq_length
        self.pooling: str = pooling.lower()

        if self.pooling not in ("mean", "max", "last"):
            raise ValueError(
                f"Unsupported pooling mode: '{pooling}'. Supported modes are 'mean', 'max', 'last'."
            )

        # Word Embedding Layer: zero gradients for padding index
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_dim,
            padding_idx=padding_idx,
        )
        self.embed_dropout = nn.Dropout(p=dropout_p)

        # Bidirectional LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        # Bidirectional output dimension
        self.lstm_out_dim: int = hidden_dim * 2

        # Projection Head: 2-layer MLP matching ImageEncoder architecture
        self.projection_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.BatchNorm1d(self.lstm_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.lstm_out_dim, embed_dim),
        )

    def _validate_input(self, captions: torch.Tensor) -> None:
        """
        Validate input tensor rank, dtype, and vocabulary boundary constraints.

        Args:
            captions: Input caption token tensor.

        Raises:
            TypeError: If input is not a torch.Tensor or has non-integer dtype.
            ValueError: If input is not 2D or contains token IDs outside [0, vocab_size - 1].
        """
        if not isinstance(captions, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, but received {type(captions).__name__}")
        if captions.ndim != 2:
            raise ValueError(
                f"Expected 2D input tensor [batch_size, seq_len], "
                f"but received tensor with {captions.ndim} dimension(s): shape={tuple(captions.shape)}"
            )
        if captions.dtype not in (torch.int64, torch.long):
            raise TypeError(
                f"Expected integer token tensor of dtype torch.long or torch.int64, "
                f"but received dtype {captions.dtype}"
            )
        if (captions < 0).any() or (captions >= self.vocab_size).any():
            min_val = captions.min().item()
            max_val = captions.max().item()
            raise ValueError(
                f"Token IDs out of vocabulary bounds [0, {self.vocab_size - 1}]. "
                f"Found min ID {min_val}, max ID {max_val}."
            )

    def extract_features(self, captions: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate temporal recurrent feature representations from the BiLSTM backbone.

        Args:
            captions: Caption token ID tensor [B, seq_len] with values in [0, vocab_size - 1].

        Returns:
            Recurrent feature tensor [B, seq_len, hidden_dim * 2] (e.g. [B, 32, 256]).
        """
        self._validate_input(captions)
        embedded = self.embed_dropout(self.embedding(captions))
        lstm_out, _ = self.lstm(embedded)
        return lstm_out

    def pool_features(self, lstm_out: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Pool recurrent temporal representations across the sequence dimension using padding-aware masking.

        Args:
            lstm_out: Temporal feature tensor [B, seq_len, lstm_out_dim].
            captions: Original token ID tensor [B, seq_len] used to identify <PAD> indices.

        Returns:
            Aggregated sequence representation vector [B, lstm_out_dim].
        """
        if self.pooling == "mean":
            # Mask out <PAD> positions to avoid diluting semantic representations
            mask = (captions != self.padding_idx).unsqueeze(-1).float()  # [B, seq_len, 1]
            summed = (lstm_out * mask).sum(dim=1)  # [B, lstm_out_dim]
            lengths = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            return summed / lengths

        elif self.pooling == "max":
            pad_mask = (captions == self.padding_idx).unsqueeze(-1)  # [B, seq_len, 1]
            masked_out = lstm_out.masked_fill(pad_mask, -1e9)
            max_pooled = torch.max(masked_out, dim=1)[0]
            # Handle edge case where sequence consists entirely of padding tokens
            all_pad = (captions != self.padding_idx).sum(dim=1, keepdim=True) == 0  # [B, 1]
            return max_pooled.masked_fill(all_pad, 0.0)

        elif self.pooling == "last":
            lengths = (captions != self.padding_idx).sum(dim=1)  # [B]
            last_indices = (lengths - 1).clamp(min=0)  # [B]
            batch_indices = torch.arange(captions.size(0), device=captions.device)
            last_pooled = lstm_out[batch_indices, last_indices, :]
            all_pad = (lengths == 0).unsqueeze(-1)
            return last_pooled.masked_fill(all_pad, 0.0)

        raise ValueError(f"Unsupported pooling mode: '{self.pooling}'")

    def forward(self, captions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass mapping input token sequences to shared embedding space.

        Args:
            captions: Caption token ID tensor [B, seq_len] (e.g. [B, 32]).
            normalize: If True, applies L2 unit normalization along embedding dimension.

        Returns:
            Text embeddings [B, embed_dim] (norm = 1.0 when normalize=True).
        """
        self._validate_input(captions)
        lstm_out = self.extract_features(captions)
        pooled = self.pool_features(lstm_out, captions)
        projected = self.projection_head(pooled)

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

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size accepted by the encoder.

        Returns:
            Vocabulary size integer (e.g. 158).
        """
        return self.vocab_size
