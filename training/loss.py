"""
WildMatch AI - Member 3: Contrastive Loss Module (`training/loss.py`)

Implements:
1. Symmetric Contrastive Loss (InfoNCE / NT-Xent) for cross-modal image-text alignment:
   L = (L_image->text + L_text->image) / 2
2. Species-Aware Multi-Positive Contrastive Loss (Supervised Multi-Positive InfoNCE):
   Accommodates multiple in-batch samples sharing the same biological species label,
   mitigating the false negative penalty inherent in standard diagonal InfoNCE.
3. Auxiliary Classification Loss:
   Standard cross-entropy loss over species classes for visual and textual representations.
4. Unified WildMatch Joint Loss:
   L_total = L_contrastive + lambda_aux * L_aux
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class SymmetricContrastiveLoss(nn.Module):
    """
    Symmetric Image-Text Contrastive Loss (Bidirectional InfoNCE / NT-Xent).

    Computes cosine similarity between unit L2-normalized image embeddings [B, D]
    and text embeddings [B, D], scales by temperature tau (or logit_scale),
    and computes categorical cross-entropy loss in both directions:
        L_i2t = CrossEntropy(logits_per_image, target_indices)
        L_t2i = CrossEntropy(logits_per_text, target_indices)
        L_symmetric = (L_i2t + L_t2i) / 2

    Supports backpropagation through both image and text representation spaces,
    as well as learnable logit_scale parameters.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_scale: bool = False,
        max_temperature: float = 100.0,
    ) -> None:
        """
        Args:
            temperature: Default temperature parameter tau > 0 (default: 0.07).
            learnable_scale: If True, maintains an internal learnable logit_scale parameter.
            max_temperature: Upper clamp bound for temperature scaling (default: 100.0).
        """
        super().__init__()
        self.default_temperature = temperature
        self.max_temperature = max_temperature
        if learnable_scale:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
            )
        else:
            self.register_parameter("logit_scale", None)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute symmetric contrastive loss.

        Args:
            image_embeddings: FloatTensor of shape [B, D].
            text_embeddings: FloatTensor of shape [B, D].
            logit_scale: Optional positive scale factor (exp(logit_scale) or 1 / tau).
                         If None, uses self.logit_scale or 1.0 / default_temperature.
            normalize: If True, applies unit L2 normalization to both input tensors.

        Returns:
            Scalar loss tensor supporting autograd.
        """
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding tensors [B, D], got {tuple(image_embeddings.shape)} "
                f"and {tuple(text_embeddings.shape)}"
            )
        if image_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError(
                f"Batch size mismatch: {image_embeddings.size(0)} images vs "
                f"{text_embeddings.size(0)} texts"
            )
        if image_embeddings.size(1) != text_embeddings.size(1):
            raise ValueError(
                f"Embedding dimension mismatch: {image_embeddings.size(1)} vs "
                f"{text_embeddings.size(1)}"
            )

        batch_size = image_embeddings.size(0)
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Determine scale factor
        if logit_scale is not None:
            scale = logit_scale.clamp(max=self.max_temperature)
        elif self.logit_scale is not None:
            scale = self.logit_scale.exp().clamp(max=self.max_temperature)
        else:
            scale = torch.tensor(1.0 / self.default_temperature, device=image_embeddings.device)

        # Pairwise cosine similarity matrix [B, B]
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        logits_per_image = scale * similarity
        logits_per_text = logits_per_image.t()

        # Diagonal target indices [0, 1, ..., B-1]
        labels = torch.arange(batch_size, device=image_embeddings.device, dtype=torch.long)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2.0


class SpeciesAwareMultiPositiveContrastiveLoss(nn.Module):
    """
    Species-Aware Multi-Positive Supervised Contrastive Loss.

    In a small-vocabulary or wildlife dataset, mini-batches frequently contain
    multiple items from the same species class (e.g. 4 Tigers out of 16 samples).
    Standard diagonal InfoNCE penalizes these co-species pairs as false negatives,
    disrupting intra-species semantic clustering.

    This loss treats all pairs sharing the same species label as positives:
        L_i = - (1 / |P(i)|) * sum_{p in P(i)} log [ exp(sim(i, p) / tau) / sum_k exp(sim(i, k) / tau) ]
    symmetrically applied across visual and textual modalities.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        max_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        self.default_temperature = temperature
        self.max_temperature = max_temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        species_indices: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            image_embeddings: [B, D]
            text_embeddings: [B, D]
            species_indices: [B] integer class labels in [0, num_classes-1]
            logit_scale: Optional scalar temperature scale
            normalize: Apply unit L2 normalization
        """
        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        batch_size = image_embeddings.size(0)
        if logit_scale is not None:
            scale = logit_scale.clamp(max=self.max_temperature)
        else:
            scale = torch.tensor(1.0 / self.default_temperature, device=image_embeddings.device)

        # Similarity matrix scaled by temperature
        sim = scale * torch.matmul(image_embeddings, text_embeddings.t())  # [B, B]

        # Multi-positive binary mask [B, B] where True indicates identical species
        species_indices = species_indices.view(-1)
        mask = (species_indices.unsqueeze(0) == species_indices.unsqueeze(1)).float()  # [B, B]

        # Log-softmax along candidate dimension
        log_prob_i2t = F.log_softmax(sim, dim=1)
        log_prob_t2i = F.log_softmax(sim.t(), dim=1)

        # Mean log-probability over positive pairs per sample
        pos_counts_i2t = mask.sum(dim=1).clamp(min=1.0)
        pos_counts_t2i = mask.sum(dim=0).clamp(min=1.0)

        loss_i2t = -(mask * log_prob_i2t).sum(dim=1) / pos_counts_i2t
        loss_t2i = -(mask * log_prob_t2i).sum(dim=1) / pos_counts_t2i

        return (loss_i2t.mean() + loss_t2i.mean()) / 2.0


class AuxiliaryClassificationLoss(nn.Module):
    """
    Auxiliary Cross-Entropy Classification Loss for visual and textual representations.

    Regularizes the dual-encoder latent space by predicting biological species classes
    from both visual and textual embeddings.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        image_logits: torch.Tensor,
        text_logits: torch.Tensor,
        target_species_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_aux_loss: (loss_image + loss_text) / 2
            loss_image: CrossEntropy on image logits
            loss_text: CrossEntropy on text logits
        """
        loss_img = F.cross_entropy(
            image_logits, target_species_idx, label_smoothing=self.label_smoothing
        )
        loss_txt = F.cross_entropy(
            text_logits, target_species_idx, label_smoothing=self.label_smoothing
        )
        total = (loss_img + loss_txt) / 2.0
        return total, loss_img, loss_txt


class WildMatchLoss(nn.Module):
    """
    Unified WildMatch Joint Training Objective.

    Combines:
    - Contrastive Loss (standard symmetric InfoNCE or species-aware multi-positive)
    - Optional auxiliary classification loss weighted by lambda_aux
    """

    def __init__(
        self,
        contrastive_type: str = "multi_positive",
        temperature: float = 0.07,
        lambda_aux: float = 0.5,
    ) -> None:
        super().__init__()
        self.contrastive_type = contrastive_type
        self.lambda_aux = lambda_aux

        self.symmetric_loss = SymmetricContrastiveLoss(temperature=temperature)
        self.multi_pos_loss = SpeciesAwareMultiPositiveContrastiveLoss(temperature=temperature)
        self.aux_loss = AuxiliaryClassificationLoss()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        species_indices: Optional[torch.Tensor] = None,
        image_class_logits: Optional[torch.Tensor] = None,
        text_class_logits: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes joint training loss and returns a detailed breakdown dictionary.
        """
        if self.contrastive_type == "multi_positive" and species_indices is not None:
            contrastive = self.multi_pos_loss(
                image_embeddings,
                text_embeddings,
                species_indices=species_indices,
                logit_scale=logit_scale,
            )
        else:
            contrastive = self.symmetric_loss(
                image_embeddings, text_embeddings, logit_scale=logit_scale
            )

        total = contrastive
        aux_total = torch.tensor(0.0, device=image_embeddings.device)
        aux_img = torch.tensor(0.0, device=image_embeddings.device)
        aux_txt = torch.tensor(0.0, device=image_embeddings.device)

        if (
            self.lambda_aux > 0.0
            and species_indices is not None
            and image_class_logits is not None
            and text_class_logits is not None
        ):
            aux_total, aux_img, aux_txt = self.aux_loss(
                image_class_logits, text_class_logits, species_indices
            )
            total = total + self.lambda_aux * aux_total

        return {
            "total_loss": total,
            "contrastive_loss": contrastive,
            "aux_total_loss": aux_total,
            "aux_image_loss": aux_img,
            "aux_text_loss": aux_txt,
        }

