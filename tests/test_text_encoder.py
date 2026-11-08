# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import unittest
import sys
from pathlib import Path

# Ensure WildMatch_AI package root and parent workspace are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WILDMATCH_ROOT = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(WILDMATCH_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from WildMatch_AI.models.text_encoder import TextEncoder
    from WildMatch_AI.preprocessing.text_preprocessing import Vocabulary, tokenize_caption
except ImportError:
    from models.text_encoder import TextEncoder
    from preprocessing.text_preprocessing import Vocabulary, tokenize_caption


class TestTextEncoder(unittest.TestCase):
    """
    Comprehensive unit test suite for the BiLSTM TextEncoder module.
    """

    def setUp(self) -> None:
        """Set fixed seed for determinism in unit tests."""
        torch.manual_seed(42)

    def test_forward_pass_batch_sizes(self) -> None:
        """
        Verify forward pass produces [B, embed_dim] across batch sizes 1, 4, 16.
        In eval() mode, BatchNorm handles single-sample batches properly.
        """
        model = TextEncoder(vocab_size=158, embed_dim=256)
        model.eval()

        for batch_size in (1, 4, 16):
            with torch.no_grad():
                x = torch.randint(0, 158, (batch_size, 32), dtype=torch.long)
                out = model(x)
                self.assertEqual(
                    out.shape,
                    (batch_size, 256),
                    f"Expected shape ({batch_size}, 256), got {out.shape}",
                )
                self.assertEqual(out.dtype, torch.float32)

    def test_configurable_embed_dimensions(self) -> None:
        """
        Verify output dimensionality matches configurable embed_dim (128, 256, 512)
        and getter methods return correct properties.
        """
        batch_size = 4
        x = torch.randint(0, 158, (batch_size, 32), dtype=torch.long)

        for embed_dim in (128, 256, 512):
            model = TextEncoder(vocab_size=158, embed_dim=embed_dim)
            model.eval()
            with torch.no_grad():
                out = model(x)
                self.assertEqual(
                    out.shape,
                    (batch_size, embed_dim),
                    f"Expected output dimension {embed_dim}, got {out.shape[1]}",
                )
                self.assertEqual(model.get_embedding_dim(), embed_dim)
                self.assertEqual(model.get_vocab_size(), 158)

    def test_l2_normalization_flag(self) -> None:
        """
        Verify L2 unit normalization behavior:
        - normalize=True: norm == 1.0 +/- 1e-5.
        - normalize=False: raw projection outputs (norm is unconstrained).
        """
        model = TextEncoder(vocab_size=158, embed_dim=256)
        model.eval()
        x = torch.randint(0, 158, (8, 32), dtype=torch.long)

        with torch.no_grad():
            # Normalized embeddings
            out_norm = model(x, normalize=True)
            norms = torch.norm(out_norm, p=2, dim=-1)
            expected_ones = torch.ones_like(norms)
            self.assertTrue(
                torch.allclose(norms, expected_ones, atol=1e-5),
                f"L2 norms deviate from 1.0: min={norms.min().item()}, max={norms.max().item()}",
            )

            # Unnormalized embeddings
            out_raw = model(x, normalize=False)
            raw_norms = torch.norm(out_raw, p=2, dim=-1)
            self.assertFalse(
                torch.allclose(raw_norms, expected_ones, atol=1e-3),
                "Unnormalized outputs unexpectedly have unit norm",
            )

    def test_sequence_feature_extraction(self) -> None:
        """
        Verify intermediate feature extraction produces [B, seq_len, hidden_dim * 2]
        for standard input [B, 32].
        """
        hidden_dim = 128
        model = TextEncoder(vocab_size=158, hidden_dim=hidden_dim, embed_dim=256)
        model.eval()
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, 158, (batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            features = model.extract_features(x)
            self.assertEqual(
                features.shape,
                (batch_size, seq_len, hidden_dim * 2),
                f"Expected feature map shape ({batch_size}, {seq_len}, {hidden_dim * 2}), got {features.shape}",
            )

    def test_masked_pooling_padding_invariance(self) -> None:
        """
        Verify that sequence pooling masks out <PAD> (index 0) tokens:
        - Feature variations at padded token positions do not alter the pooled mean representation.
        - All-padding edge cases do not result in NaN values.
        - Alternative pooling strategies ('max', 'last') execute without error.
        """
        model = TextEncoder(vocab_size=158, hidden_dim=64, embed_dim=128, pooling="mean")
        model.eval()

        captions = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)  # [1, 5]
        dummy_lstm_out = torch.ones(1, 5, 128)

        # Output with zeros at padding positions
        pooled_1 = model.pool_features(dummy_lstm_out, captions)

        # Mutate padding positions to large random noise
        dummy_corrupted = dummy_lstm_out.clone()
        dummy_corrupted[:, 3:, :] = 999.0
        pooled_2 = model.pool_features(dummy_corrupted, captions)

        # Masked pooling must ignore positions where caption == 0
        self.assertTrue(
            torch.allclose(pooled_1, pooled_2, atol=1e-6),
            "Padding positions corrupted the pooled mean representation",
        )

        # Verify mean pooling is exact average of first 3 tokens: (1 + 1 + 1) / 3 = 1.0
        self.assertTrue(torch.allclose(pooled_1, torch.ones_like(pooled_1), atol=1e-6))

        # Test edge case: all-padding sequence produces zeros without NaN
        all_pad_captions = torch.zeros((2, 32), dtype=torch.long)
        all_pad_lstm = torch.randn(2, 32, 128)
        pooled_pad = model.pool_features(all_pad_lstm, all_pad_captions)
        self.assertFalse(torch.isnan(pooled_pad).any(), "All-pad sequence produced NaN in pooled features")
        self.assertTrue(torch.allclose(pooled_pad, torch.zeros_like(pooled_pad), atol=1e-6))

        # Test 'max' pooling
        model_max = TextEncoder(vocab_size=158, hidden_dim=64, embed_dim=128, pooling="max")
        pooled_max = model_max.pool_features(dummy_lstm_out, captions)
        self.assertEqual(pooled_max.shape, (1, 128))
        self.assertFalse(torch.isnan(pooled_max).any())

        # Test 'last' pooling
        model_last = TextEncoder(vocab_size=158, hidden_dim=64, embed_dim=128, pooling="last")
        pooled_last = model_last.pool_features(dummy_lstm_out, captions)
        self.assertEqual(pooled_last.shape, (1, 128))
        self.assertFalse(torch.isnan(pooled_last).any())

    def test_gradient_backpropagation(self) -> None:
        """
        Verify backward pass computes valid gradients for embedding, BiLSTM, BatchNorm,
        and Linear projection parameters without NaN or Inf.
        Verify padding token (index 0) receives zero gradient.
        """
        model = TextEncoder(vocab_size=158, word_embed_dim=64, hidden_dim=64, embed_dim=128)
        model.train()

        # Batch of 4 captions containing some padding tokens
        captions = torch.randint(1, 158, (4, 32), dtype=torch.long)
        captions[:, 20:] = 0  # Right pad with 0

        embeddings = model(captions, normalize=True)
        loss = embeddings.sum()
        loss.backward()

        unassigned_grads = []
        nan_or_inf_grads = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    unassigned_grads.append(name)
                elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_or_inf_grads.append(name)

        self.assertEqual(
            len(unassigned_grads),
            0,
            f"Parameters without gradients: {unassigned_grads}",
        )
        self.assertEqual(
            len(nan_or_inf_grads),
            0,
            f"Parameters with NaN or Inf gradients: {nan_or_inf_grads}",
        )

        # Padding index row (0) in embedding layer must have zero gradient
        pad_grad = model.embedding.weight.grad[0]
        self.assertTrue(
            torch.all(pad_grad == 0),
            "Padding index (0) embedding received non-zero gradient",
        )

    def test_eval_and_train_modes(self) -> None:
        """
        Verify model properly toggles between train() and eval() modes.
        In eval mode, dropout is deactivated and BatchNorm uses running statistics.
        """
        model = TextEncoder(vocab_size=158, embed_dim=256, dropout_p=0.5)

        # Train mode
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.embed_dropout.training)
        self.assertTrue(model.projection_head[3].training)  # Projection dropout

        # Eval mode
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.embed_dropout.training)
        self.assertFalse(model.projection_head[3].training)

        # Eval mode determinism
        x = torch.randint(0, 158, (2, 32), dtype=torch.long)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            self.assertTrue(
                torch.allclose(out1, out2, atol=1e-7),
                "Eval mode forward pass should be fully deterministic",
            )

    def test_invalid_inputs_raise_error(self) -> None:
        """
        Verify invalid input dimensions, data types, or out-of-vocabulary indices raise exceptions.
        """
        model = TextEncoder(vocab_size=158, embed_dim=256)
        model.eval()

        # Non-tensor input
        with self.assertRaises(TypeError):
            model("not a tensor")  # type: ignore

        # 1D tensor [32]
        x_1d = torch.randint(0, 158, (32,), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x_1d)

        with self.assertRaises(ValueError):
            model.extract_features(x_1d)

        # 3D tensor [2, 32, 1]
        x_3d = torch.randint(0, 158, (2, 32, 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x_3d)

        # Float tensor [2, 32]
        x_float = torch.rand(2, 32, dtype=torch.float32)
        with self.assertRaises(TypeError):
            model(x_float)

        # Out of bounds token index (>= 158)
        x_oob_high = torch.tensor([[10, 158]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x_oob_high)

        # Negative token index (< 0)
        x_oob_low = torch.tensor([[10, -1]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x_oob_low)

        # Invalid pooling strategy in constructor
        with self.assertRaises(ValueError):
            TextEncoder(vocab_size=158, pooling="invalid_pool")

    def test_member1_vocabulary_integration(self) -> None:
        """
        Verify real Member 1 vocabulary loading, tokenization, and numericalization
        for realistic wildlife captions produces valid [B, 256] unit-normalized embeddings.
        """
        vocab_path = PROJECT_ROOT / "WildMatch_AI" / "dataset" / "vocabulary.json"
        if not vocab_path.exists():
            self.skipTest(f"Vocabulary file not found at {vocab_path}")

        vocab = Vocabulary.load_json(str(vocab_path))
        self.assertEqual(len(vocab.word2idx), 158)

        sample_captions = [
            "a tiger resting in water",
            "asian elephant grazing on hillside",
            "two lions in grass",
        ]

        token_id_tensors = []
        for text in sample_captions:
            tokens = tokenize_caption(text)
            ids = vocab.numericalize(tokens, add_special_tokens=True, max_len=32, pad=True)
            self.assertEqual(len(ids), 32)
            token_id_tensors.append(torch.tensor(ids, dtype=torch.long))

        batch_captions = torch.stack(token_id_tensors, dim=0)  # [3, 32]
        self.assertEqual(batch_captions.shape, (3, 32))

        model = TextEncoder(vocab_size=len(vocab.word2idx), embed_dim=256)
        model.eval()

        with torch.no_grad():
            embeddings = model(batch_captions, normalize=True)
            self.assertEqual(embeddings.shape, (3, 256))
            norms = torch.norm(embeddings, p=2, dim=-1)
            self.assertTrue(
                torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                f"L2 norms deviate from 1.0: {norms}",
            )


if __name__ == "__main__":
    unittest.main()
