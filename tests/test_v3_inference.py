"""
Unit tests for WildMatch AI Version 3 Architecture and Inference Pipeline.
"""

import unittest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wildmatch_v3 import (
    WildMatchV3DualEncoder,
    WildMatchV3ImageEncoder,
    WildMatchV3TextEncoder,
    WildMatchV3SpeciesHeads,
    SPECIES_CLASSES,
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


class TestWildMatchV3(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dual_ckpt = PROJECT_ROOT / "models" / "wildmatch_v3_dual_encoder.pth"
        self.heads_ckpt = PROJECT_ROOT / "models" / "wildmatch_v3_species_heads.pth"

    def test_checkpoint_loading(self):
        """Verify checkpoint loads cleanly without missing or unexpected keys."""
        model = load_model(self.dual_ckpt)
        self.assertIsInstance(model, WildMatchV3DualEncoder)
        self.assertEqual(model.embed_dim, 256)

        heads = load_species_heads(self.heads_ckpt)
        self.assertIsInstance(heads, WildMatchV3SpeciesHeads)
        self.assertEqual(heads.species, SPECIES_CLASSES)

    def test_image_encoder_forward(self):
        """Verify image encoder output shape and unit L2 norm."""
        img_enc = WildMatchV3ImageEncoder(embed_dim=256)
        img_enc.eval()
        dummy_imgs = torch.rand(3, 3, 224, 224)
        out = img_enc(dummy_imgs, normalize=True)
        self.assertEqual(out.shape, (3, 256))
        norms = torch.norm(out, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))

    def test_text_encoder_forward(self):
        """Verify text encoder output shape and unit L2 norm."""
        txt_enc = WildMatchV3TextEncoder(vocab_size=158, embed_dim=256)
        txt_enc.eval()
        dummy_tokens = torch.randint(0, 158, (3, 32))
        out = txt_enc(dummy_tokens, normalize=True)
        self.assertEqual(out.shape, (3, 256))
        norms = torch.norm(out, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(3), atol=1e-5))

    def test_tokenization(self):
        """Verify caption tokenization produces [1, 32] LongTensor with BOS and EOS."""
        tokens = tokenize_caption("a tiger resting near water")
        self.assertEqual(tokens.shape, (1, 32))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens[0, 0].item(), 2)  # BOS
        # EOS should be present
        self.assertIn(3, tokens[0].tolist())

    def test_similarity_computation(self):
        """Verify compute_similarity matrix dimensions."""
        model = load_model(self.dual_ckpt)
        imgs = torch.rand(2, 3, 224, 224)
        tokens = torch.randint(0, 158, (4, 32))

        img_embs = embed_image(model, imgs)
        txt_embs = embed_text(model, tokens)

        sim = compute_similarity(img_embs, txt_embs)
        self.assertEqual(sim.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()

