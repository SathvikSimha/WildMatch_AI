# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn.functional as F
import math
import sys
import unittest
import json
from pathlib import Path

# Ensure WildMatch_AI package root and parent workspace are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WILDMATCH_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(WILDMATCH_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from WildMatch_AI.models.dual_encoder import DualEncoder
    from WildMatch_AI.models.image_encoder import ImageEncoder
    from WildMatch_AI.models.text_encoder import TextEncoder
    from WildMatch_AI.preprocessing.text_preprocessing import Vocabulary, tokenize_caption
except ImportError:
    from models.dual_encoder import DualEncoder
    from models.image_encoder import ImageEncoder
    from models.text_encoder import TextEncoder
    from preprocessing.text_preprocessing import Vocabulary, tokenize_caption


class TestDualEncoder(unittest.TestCase):
    """
    Comprehensive unit test suite for the DualEncoder multimodal architecture.
    """

    def setUp(self) -> None:
        """Set fixed seed for determinism in unit tests."""
        torch.manual_seed(42)

    def test_forward_pass_shapes_and_types(self) -> None:
        """
        Verify forward(images, captions) returns (image_embeds, text_embeds, scale)
        with expected shapes, types, and unit L2 norms across batch sizes 2, 4, 8.
        """
        model = DualEncoder(embed_dim=256)
        model.eval()

        for batch_size in (2, 4, 8):
            with torch.no_grad():
                imgs = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
                caps = torch.randint(0, 158, (batch_size, 32), dtype=torch.long)
                img_emb, txt_emb, scale = model(imgs, caps)

                self.assertEqual(img_emb.shape, (batch_size, 256))
                self.assertEqual(txt_emb.shape, (batch_size, 256))
                self.assertEqual(img_emb.dtype, torch.float32)
                self.assertEqual(txt_emb.dtype, torch.float32)

                # Scale should be a scalar tensor > 0
                self.assertEqual(scale.ndim, 0)
                self.assertGreater(scale.item(), 0.0)

                # Both embeddings should have unit L2 norm
                img_norms = torch.norm(img_emb, p=2, dim=-1)
                txt_norms = torch.norm(txt_emb, p=2, dim=-1)

                expected_norms = torch.ones(batch_size)
                torch.testing.assert_close(
                    img_norms, expected_norms, atol=1e-5, rtol=1e-5
                )
                torch.testing.assert_close(
                    txt_norms, expected_norms, atol=1e-5, rtol=1e-5
                )

    def test_unimodal_encode_endpoints(self) -> None:
        """
        Verify independent unimodal encoding endpoints encode_image and encode_text
        support normalized and unnormalized feature representations.
        """
        model = DualEncoder(embed_dim=256)
        model.eval()

        batch_size = 4
        imgs = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
        caps = torch.randint(0, 158, (batch_size, 32), dtype=torch.long)

        with torch.no_grad():
            # Normalized image encoding
            norm_img = model.encode_image(imgs, normalize=True)
            self.assertEqual(norm_img.shape, (batch_size, 256))
            img_norms = torch.norm(norm_img, p=2, dim=-1)
            torch.testing.assert_close(img_norms, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

            # Raw unnormalized image encoding
            raw_img = model.encode_image(imgs, normalize=False)
            self.assertEqual(raw_img.shape, (batch_size, 256))

            # Normalized text encoding
            norm_txt = model.encode_text(caps, normalize=True)
            self.assertEqual(norm_txt.shape, (batch_size, 256))
            txt_norms = torch.norm(norm_txt, p=2, dim=-1)
            torch.testing.assert_close(txt_norms, torch.ones(batch_size), atol=1e-5, rtol=1e-5)

            # Raw unnormalized text encoding
            raw_txt = model.encode_text(caps, normalize=False)
            self.assertEqual(raw_txt.shape, (batch_size, 256))

    def test_similarity_computation_square_and_rectangular(self) -> None:
        """
        Verify compute_similarity calculates cosine similarity matrices for both
        square batches (B_img == B_txt) and rectangular batches (B_img != B_txt),
        with values strictly bounded in [-1.0, 1.0].
        """
        model = DualEncoder(embed_dim=256)
        model.eval()

        with torch.no_grad():
            # Square batch (4 images, 4 captions)
            square_imgs = torch.rand(4, 3, 224, 224)
            square_caps = torch.randint(0, 158, (4, 32))
            sim_square = model.compute_similarity(square_imgs, square_caps)
            self.assertEqual(sim_square.shape, (4, 4))
            self.assertTrue(torch.all(sim_square >= -1.0001))
            self.assertTrue(torch.all(sim_square <= 1.0001))

            # Rectangular batch (3 images, 5 captions)
            rect_imgs = torch.rand(3, 3, 224, 224)
            rect_caps = torch.randint(0, 158, (5, 32))
            sim_rect = model.compute_similarity(rect_imgs, rect_caps)
            self.assertEqual(sim_rect.shape, (3, 5))
            self.assertTrue(torch.all(sim_rect >= -1.0001))
            self.assertTrue(torch.all(sim_rect <= 1.0001))

    def test_temperature_scaled_logits(self) -> None:
        """
        Verify get_logits returns (logits_per_image, logits_per_text) with proper
        transposition symmetry and scaling factor.
        """
        model = DualEncoder(embed_dim=256, init_temperature=0.07)
        model.eval()

        b_img, b_txt = 3, 5
        imgs = torch.rand(b_img, 3, 224, 224)
        caps = torch.randint(0, 158, (b_txt, 32))

        with torch.no_grad():
            logits_per_image, logits_per_text = model.get_logits(imgs, caps)
            self.assertEqual(logits_per_image.shape, (b_img, b_txt))
            self.assertEqual(logits_per_text.shape, (b_txt, b_img))

            # Check transpose symmetry
            torch.testing.assert_close(logits_per_image.t(), logits_per_text)

            # Check scaling factor consistency
            sim = model.compute_similarity(imgs, caps)
            scale = model.get_scale()
            expected_logits = scale * sim
            torch.testing.assert_close(logits_per_image, expected_logits)

    def test_learnable_logit_scale_initialization_and_clamping(self) -> None:
        """
        Verify logit_scale parameter is initialized to ln(1/init_temperature),
        requires gradients, and get_scale clamps to max_temperature.
        """
        init_temp = 0.07
        max_temp = 100.0
        model = DualEncoder(init_temperature=init_temp, max_temperature=max_temp)

        # Check parameter initialization
        expected_log_scale = math.log(1.0 / init_temp)
        self.assertAlmostEqual(model.logit_scale.item(), expected_log_scale, places=4)
        self.assertTrue(model.logit_scale.requires_grad)

        # Initial effective temperature property
        self.assertAlmostEqual(model.temperature, init_temp, places=4)

        # Test upper clamp behavior
        with torch.no_grad():
            model.logit_scale.fill_(15.0)  # exp(15) ~ 3.26e6 >> 100.0
            clamped_scale = model.get_scale()
            self.assertAlmostEqual(clamped_scale.item(), max_temp, places=3)
            self.assertAlmostEqual(model.temperature, 1.0 / max_temp, places=4)

    def test_configurable_embed_dimensions(self) -> None:
        """
        Verify DualEncoder functions across multiple embedding dimensions (128, 256, 512)
        and raises ValueError when sub-encoders have mismatched embed_dim.
        """
        for dim in (128, 256, 512):
            model = DualEncoder(embed_dim=dim)
            model.eval()
            self.assertEqual(model.get_embedding_dim(), dim)

            imgs = torch.rand(2, 3, 224, 224)
            caps = torch.randint(0, 158, (2, 32))
            with torch.no_grad():
                img_emb, txt_emb, _ = model(imgs, caps)
                self.assertEqual(img_emb.shape, (2, dim))
                self.assertEqual(txt_emb.shape, (2, dim))

        # Test sub-encoder dimension mismatch raises ValueError
        img_enc = ImageEncoder(embed_dim=128)
        txt_enc = TextEncoder(embed_dim=256)

        with self.assertRaises(ValueError):
            DualEncoder(image_encoder=img_enc, embed_dim=256)

        with self.assertRaises(ValueError):
            DualEncoder(text_encoder=txt_enc, embed_dim=128)

    def test_gradient_backpropagation_end_to_end(self) -> None:
        """
        Verify end-to-end backpropagation through both encoders and logit_scale
        using symmetric contrastive cross-entropy loss without NaN or Inf.
        """
        model = DualEncoder(embed_dim=256)
        model.train()

        batch_size = 4
        imgs = torch.rand(batch_size, 3, 224, 224, requires_grad=False)
        caps = torch.randint(0, 158, (batch_size, 32), dtype=torch.long)
        targets = torch.arange(batch_size, dtype=torch.long)

        logits_per_image, logits_per_text = model.get_logits(imgs, caps)

        # Symmetric contrastive loss (Member 3 InfoNCE structure)
        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2.0

        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())

        loss.backward()

        # Check logit_scale gradient
        self.assertIsNotNone(model.logit_scale.grad)
        self.assertFalse(torch.isnan(model.logit_scale.grad).item())
        self.assertFalse(torch.isinf(model.logit_scale.grad).item())

        # Check ImageEncoder backbone & projection gradients
        for name, param in model.image_encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"ImageEncoder param {name} missing grad")
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    f"ImageEncoder param {name} contains NaN grad",
                )
                self.assertFalse(
                    torch.isinf(param.grad).any().item(),
                    f"ImageEncoder param {name} contains Inf grad",
                )

        # Check TextEncoder embedding, LSTM, & projection gradients
        for name, param in model.text_encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"TextEncoder param {name} missing grad")
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    f"TextEncoder param {name} contains NaN grad",
                )
                self.assertFalse(
                    torch.isinf(param.grad).any().item(),
                    f"TextEncoder param {name} contains Inf grad",
                )

    def test_eval_and_train_modes(self) -> None:
        """
        Verify train() and eval() switch state across all submodules,
        and batch size 1 forward evaluation works smoothly in eval mode.
        """
        model = DualEncoder(embed_dim=256)

        # Switch to eval mode
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.image_encoder.training)
        self.assertFalse(model.text_encoder.training)

        # Batch size 1 forward in eval mode
        with torch.no_grad():
            img_b1 = torch.rand(1, 3, 224, 224)
            cap_b1 = torch.randint(0, 158, (1, 32))
            img_emb, txt_emb, _ = model(img_b1, cap_b1)
            self.assertEqual(img_emb.shape, (1, 256))
            self.assertEqual(txt_emb.shape, (1, 256))

        # Switch back to train mode
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.image_encoder.training)
        self.assertTrue(model.text_encoder.training)

    def test_real_member1_pipeline_integration(self) -> None:
        """
        Verify integration with Member 1's vocabulary.json and tokenization pipeline.
        """
        vocab_path = WILDMATCH_ROOT / "dataset" / "vocabulary.json"
        if not vocab_path.exists():
            self.skipTest(f"vocabulary.json not found at {vocab_path}")

        vocab = Vocabulary.load_json(str(vocab_path))
        sample_captions = [
            "A royal bengal tiger walks through tall grass",
            "An asian elephant grazes near a watering hole",
            "A spotted chital deer stands alert in forest",
            "A lion resting under shade of acacia tree",
        ]

        token_list = [
            torch.tensor(
                vocab.numericalize(tokenize_caption(c), add_special_tokens=True, max_len=32, pad=True),
                dtype=torch.long,
            )
            for c in sample_captions
        ]
        caption_batch = torch.stack(token_list)
        image_batch = torch.rand(len(sample_captions), 3, 224, 224)

        model = DualEncoder(embed_dim=256, text_kwargs={"vocab_size": len(vocab)})
        model.eval()

        with torch.no_grad():
            img_emb, txt_emb, scale = model(image_batch, caption_batch)
            sim = model.compute_similarity(image_batch, caption_batch)
            logits_i, logits_t = model.get_logits(image_batch, caption_batch)

            self.assertEqual(img_emb.shape, (4, 256))
            self.assertEqual(txt_emb.shape, (4, 256))
            self.assertEqual(sim.shape, (4, 4))
            self.assertEqual(logits_i.shape, (4, 4))
            self.assertEqual(logits_t.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
