"""
WildMatch AI - Version 3 Architecture and Inference Module.

Defines the exact trained Version 3 dual-encoder architecture:
- Image encoder: ImageNet-pretrained ResNet-18 backbone (512-dim features) +
                 projection MLP (512 -> 512 -> 256) + unit L2 normalization.
- Text encoder: Dense word embeddings (158 vocab -> 128-dim) +
                Bidirectional LSTM (128-dim per direction -> 256-dim) +
                padding-aware masked mean pooling +
                projection MLP (256 -> 256 -> 256) + unit L2 normalization.
- Multimodal temperature scaling: logit_scale parameter.
- Auxiliary classification heads: 256 -> 4 species heads for image and text embeddings.

For inference, ResNet-18 is instantiated directly without downloading or
initializing pretrained weights from the internet, then loaded from the
saved trained state dictionary.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Standard Species Configuration (Alphabetical Class Indexing)
# ---------------------------------------------------------------------------
SPECIES_CLASSES: List[str] = ["Asian Elephant", "Chital", "Lion", "Tiger"]
SPECIES_TO_IDX: Dict[str, int] = {species: idx for idx, species in enumerate(SPECIES_CLASSES)}
IDX_TO_SPECIES: Dict[int, str] = {idx: species for idx, species in enumerate(SPECIES_CLASSES)}

# ImageNet normalization statistics used during ResNet-18 fine-tuning
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# ResNet-18 Backbone (Pure PyTorch, Zero Remote Download Required)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock with two 3x3 convolutions and a residual shortcut.
    """
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 convolutional backbone matching the torchvision layer hierarchy
    used in WildMatch AI Version 3 training:
    conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    def _make_layer(
        self, in_planes: int, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ---------------------------------------------------------------------------
# Version 3 Image Encoder (ResNet-18 + 512 -> 512 -> 256 Projection Head)
# ---------------------------------------------------------------------------
class WildMatchV3ImageEncoder(nn.Module):
    """
    Version 3 Image Encoder for WildMatch AI.

    Extracts visual features via ResNet-18, performs adaptive average pooling to 512-d,
    projects through a 2-layer MLP (512 -> 512 -> 256), and unit L2-normalizes.
    """

    def __init__(self, embed_dim: int = 256, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = ResNet18Backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, embed_dim),
        )

    @property
    def projection_head(self) -> nn.Sequential:
        """Alias for backward compatibility."""
        return self.projection

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"Expected 4D image tensor [B, 3, H, W], got shape {tuple(x.shape)}"
            )
        features = self.backbone(x)
        pooled = self.pool(features)
        flattened = self.flatten(pooled)
        projected = self.projection(flattened)
        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map [B, 512, 7, 7] prior to pooling."""
        return self.backbone(x)

    def get_embedding_dim(self) -> int:
        return self.embed_dim


# ---------------------------------------------------------------------------
# Version 3 Text Encoder (Dense Embeddings + BiLSTM + 256 -> 256 -> 256 Head)
# ---------------------------------------------------------------------------
class WildMatchV3TextEncoder(nn.Module):
    """
    Version 3 Text Encoder for WildMatch AI.

    Encodes [B, 32] integer token sequences:
    - Word embedding: [158, 128], padding_idx=0.
    - Bidirectional LSTM: hidden_size=128 per direction -> [B, 32, 256].
    - Padding-aware masked mean pooling -> [B, 256].
    - 2-layer MLP projection: 256 -> 256 -> 256.
    - Unit L2 normalization.
    """

    def __init__(
        self,
        vocab_size: int = 158,
        word_embed_dim: int = 128,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        padding_idx: int = 0,
        dropout_p: float = 0.1,
        max_seq_length: int = 32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_dim,
            padding_idx=padding_idx,
        )
        self.embed_dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

    @property
    def projection_head(self) -> nn.Sequential:
        """Alias for backward compatibility."""
        return self.projection

    def forward(self, token_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                f"Expected 2D token tensor [B, seq_len], got shape {tuple(token_ids.shape)}"
            )

        # Boolean mask where non-padding tokens are True
        mask = (token_ids != self.padding_idx).unsqueeze(-1).float()

        # Word embedding with dropout
        emb = self.embed_dropout(self.embedding(token_ids))

        # BiLSTM contextual encoding
        lstm_out, _ = self.lstm(emb)  # [B, seq_len, 256]

        # Padding-aware masked mean pooling
        sum_masked = (lstm_out * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_masked / lens  # [B, 256]

        # Projection MLP
        projected = self.projection(pooled)
        if normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected

    def get_embedding_dim(self) -> int:
        return self.embed_dim


# ---------------------------------------------------------------------------
# Version 3 Dual Encoder (Multimodal Composition & Logit Scaling)
# ---------------------------------------------------------------------------
class WildMatchV3DualEncoder(nn.Module):
    """
    Version 3 Dual-Encoder Multimodal Architecture.

    Combines WildMatchV3ImageEncoder and WildMatchV3TextEncoder into a unified
    bimodal matching network with learnable logit scaling.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        vocab_size: int = 158,
        init_temperature: float = 0.07,
        max_temperature: float = 100.0,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_temperature = max_temperature

        self.image_encoder = WildMatchV3ImageEncoder(
            embed_dim=embed_dim, dropout_p=dropout_p
        )
        self.text_encoder = WildMatchV3TextEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim, dropout_p=dropout_p
        )
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature))
        )

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode images into [B, 256] unit embeddings."""
        return self.image_encoder(images, normalize=normalize)

    def encode_text(self, captions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode tokenized captions into [B, 256] unit embeddings."""
        return self.text_encoder(captions, normalize=normalize)

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor, normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joint multimodal forward pass.

        Returns:
            image_embeds: [B_img, 256]
            text_embeds: [B_txt, 256]
            scale: positive temperature scale factor
        """
        image_embeds = self.encode_image(images, normalize=normalize)
        text_embeds = self.encode_text(captions, normalize=normalize)
        scale = self.logit_scale.exp().clamp(max=self.max_temperature)
        return image_embeds, text_embeds, scale

    def compute_similarity(
        self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix [B_img, B_txt].
        Embeddings are assumed to be unit L2-normalized.
        """
        return torch.matmul(image_embeddings, text_embeddings.t())

    def get_logits(
        self, images: torch.Tensor, captions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric temperature-scaled logits for contrastive loss.

        Returns:
            logits_per_image: [B, B]
            logits_per_text: [B, B]
        """
        image_embeds, text_embeds, scale = self.forward(images, captions, normalize=True)
        logits_per_image = scale * torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = "models/wildmatch_v3_dual_encoder.pth",
        map_location: str = "cpu",
    ) -> "WildMatchV3DualEncoder":
        """
        Instantiate the model and load weights from the trained checkpoint.
        Does not download or initialize external weights.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            # Check models directory fallback
            alt_path = Path("models") / ckpt_path.name
            if alt_path.is_file():
                ckpt_path = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        ckpt = torch.load(ckpt_path, map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt)

        vocab_size = ckpt.get("vocab_size", 158)
        embed_dim = ckpt.get("embed_dim", 256)

        model = cls(embed_dim=embed_dim, vocab_size=vocab_size)
        model.load_state_dict(state_dict, strict=True)
        model.to(map_location)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Version 3 Auxiliary Species Classification Heads
# ---------------------------------------------------------------------------
class WildMatchV3SpeciesHeads(nn.Module):
    """
    Auxiliary image and text classification heads mapping 256-d embeddings
    to 4-class species logits.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 4,
        species_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.species = species_list or list(SPECIES_CLASSES)

        self.image_classifier = nn.Linear(embed_dim, num_classes)
        self.text_classifier = nn.Linear(embed_dim, num_classes)

    def predict_image(
        self, image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Predict species from image embeddings.

        Returns:
            predicted_indices: [B]
            predicted_labels: List of species names
            probabilities: [B, 4]
        """
        logits = self.image_classifier(image_embeddings)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        labels = [self.species[idx.item()] for idx in preds]
        return preds, labels, probs

    def predict_text(
        self, text_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Predict species from text embeddings.

        Returns:
            predicted_indices: [B]
            predicted_labels: List of species names
            probabilities: [B, 4]
        """
        logits = self.text_classifier(text_embeddings)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        labels = [self.species[idx.item()] for idx in preds]
        return preds, labels, probs

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = "models/wildmatch_v3_species_heads.pth",
        map_location: str = "cpu",
    ) -> "WildMatchV3SpeciesHeads":
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            alt_path = Path("models") / ckpt_path.name
            if alt_path.is_file():
                ckpt_path = alt_path
            else:
                raise FileNotFoundError(f"Species heads checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(ckpt_path, map_location=map_location)
        species = ckpt.get("species", SPECIES_CLASSES)
        heads = cls(embed_dim=256, num_classes=len(species), species_list=species)

        heads.image_classifier.load_state_dict(ckpt["image_classifier_state_dict"], strict=True)
        heads.text_classifier.load_state_dict(ckpt["text_classifier_state_dict"], strict=True)
        heads.to(map_location)
        heads.eval()
        return heads

