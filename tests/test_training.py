"""
Unit tests for Member 3 training and loss components:
- training/loss.py (SymmetricContrastiveLoss, SpeciesAwareMultiPositiveContrastiveLoss, AuxiliaryClassificationLoss, WildMatchLoss)
- training/train.py (build_optimizer, training utilities)
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.loss import (
    SymmetricContrastiveLoss,
    SpeciesAwareMultiPositiveContrastiveLoss,
    AuxiliaryClassificationLoss,
    WildMatchLoss,
)
from training.train import build_optimizer
from models.wildmatch_v3 import WildMatchV3DualEncoder, WildMatchV3SpeciesHeads


class TestContrastiveLosses(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.embed_dim = 256
        self.img_embeds = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)
        self.txt_embeds = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)
        self.species_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def test_symmetric_contrastive_loss_shape_and_grad(self):
        criterion = SymmetricContrastiveLoss(temperature=0.07)
        loss = criterion(self.img_embeds, self.txt_embeds)

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(self.img_embeds.grad)
        self.assertIsNotNone(self.txt_embeds.grad)
        self.assertFalse(torch.isnan(self.img_embeds.grad).any())
        self.assertFalse(torch.isnan(self.txt_embeds.grad).any())

    def test_symmetric_property(self):
        criterion = SymmetricContrastiveLoss(temperature=0.07)
        with torch.no_grad():
            loss_standard = criterion(self.img_embeds, self.txt_embeds)
            loss_transposed = criterion(self.txt_embeds, self.img_embeds)
        self.assertAlmostEqual(loss_standard.item(), loss_transposed.item(), places=5)

    def test_temperature_scaling_effect(self):
        crit_cold = SymmetricContrastiveLoss(temperature=0.01)
        crit_warm = SymmetricContrastiveLoss(temperature=0.10)
        with torch.no_grad():
            loss_cold = crit_cold(self.img_embeds, self.txt_embeds)
            loss_warm = crit_warm(self.img_embeds, self.txt_embeds)
        # Different temperatures result in different scale factors and losses
        self.assertNotEqual(loss_cold.item(), loss_warm.item())

    def test_invalid_dimensions_raise_error(self):
        criterion = SymmetricContrastiveLoss()
        with self.assertRaises(ValueError):
            criterion(torch.randn(4, 256, 1), torch.randn(4, 256))
        with self.assertRaises(ValueError):
            criterion(torch.randn(4, 256), torch.randn(3, 256))
        with self.assertRaises(ValueError):
            criterion(torch.randn(4, 256), torch.randn(4, 128))

    def test_species_aware_multi_positive_loss(self):
        criterion = SpeciesAwareMultiPositiveContrastiveLoss(temperature=0.07)
        img = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)
        txt = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)

        loss = criterion(img, txt, species_indices=self.species_indices)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(img.grad)
        self.assertIsNotNone(txt.grad)
        self.assertFalse(torch.isnan(img.grad).any())

    def test_auxiliary_classification_loss(self):
        aux_loss = AuxiliaryClassificationLoss()
        img_logits = torch.randn(self.batch_size, 4, requires_grad=True)
        txt_logits = torch.randn(self.batch_size, 4, requires_grad=True)

        total, l_img, l_txt = aux_loss(img_logits, txt_logits, self.species_indices)
        self.assertEqual(total.dim(), 0)
        self.assertTrue(torch.isfinite(total))

        total.backward()
        self.assertIsNotNone(img_logits.grad)
        self.assertIsNotNone(txt_logits.grad)

    def test_wildmatch_joint_loss(self):
        criterion = WildMatchLoss(contrastive_type="multi_positive", lambda_aux=0.5)
        img = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)
        txt = torch.randn(self.batch_size, self.embed_dim, requires_grad=True)
        img_logits = torch.randn(self.batch_size, 4, requires_grad=True)
        txt_logits = torch.randn(self.batch_size, 4, requires_grad=True)

        out = criterion(
            image_embeddings=img,
            text_embeddings=txt,
            species_indices=self.species_indices,
            image_class_logits=img_logits,
            text_class_logits=txt_logits,
        )

        self.assertIn("total_loss", out)
        self.assertIn("contrastive_loss", out)
        self.assertIn("aux_total_loss", out)
        self.assertTrue(torch.isfinite(out["total_loss"]))

        out["total_loss"].backward()
        self.assertIsNotNone(img.grad)
        self.assertIsNotNone(txt.grad)


class TestTrainingUtilities(unittest.TestCase):
    def test_build_optimizer_param_groups(self):
        model = WildMatchV3DualEncoder(embed_dim=256, vocab_size=158)
        heads = WildMatchV3SpeciesHeads(embed_dim=256, num_classes=4)

        opt = build_optimizer(model, heads, lr_backbone=1e-5, lr_heads=2e-4)
        self.assertGreaterEqual(len(opt.param_groups), 2)
        # Check that backbone group has lower lr than heads group
        lrs = [g["lr"] for g in opt.param_groups]
        self.assertIn(1e-5, lrs)
        self.assertIn(2e-4, lrs)


if __name__ == "__main__":
    unittest.main()

