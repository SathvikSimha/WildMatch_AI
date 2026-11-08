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
    from WildMatch_AI.models.image_encoder import ImageEncoder
except ImportError:
    from models.image_encoder import ImageEncoder


class TestImageEncoder(unittest.TestCase):
    """
    Comprehensive unit test suite for the Custom CNN ImageEncoder module.
    """

    def setUp(self) -> None:
        """Set fixed seed for determinism in unit tests."""
        torch.manual_seed(42)

    def test_forward_pass_batch_sizes(self) -> None:
        """
        Verify forward pass produces [B, embed_dim] across batch sizes 1, 4, 16.
        Note: For B=1 with BatchNorm in train mode, test in eval mode to prevent single-element variance error.
        """
        model = ImageEncoder(embed_dim=256)
        model.eval()

        for batch_size in (1, 4, 16):
            with torch.no_grad():
                x = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
                out = model(x)
                self.assertEqual(
                    out.shape,
                    (batch_size, 256),
                    f"Expected shape ({batch_size}, 256), got {out.shape}",
                )
                self.assertEqual(out.dtype, torch.float32)

    def test_configurable_embed_dimensions(self) -> None:
        """
        Verify output dimensionality matches configurable embed_dim (128, 256, 512).
        """
        batch_size = 4
        x = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

        for embed_dim in (128, 256, 512):
            model = ImageEncoder(embed_dim=embed_dim)
            model.eval()
            with torch.no_grad():
                out = model(x)
                self.assertEqual(
                    out.shape,
                    (batch_size, embed_dim),
                    f"Expected output dimension {embed_dim}, got {out.shape[1]}",
                )
                self.assertEqual(model.get_embedding_dim(), embed_dim)

    def test_l2_normalization_flag(self) -> None:
        """
        Verify L2 unit normalization behavior:
        - normalize=True: norm == 1.0 +/- 1e-5.
        - normalize=False: raw projection outputs (norm is not constrained to 1.0).
        """
        model = ImageEncoder(embed_dim=256)
        model.eval()
        x = torch.rand(8, 3, 224, 224, dtype=torch.float32)

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
            # Raw norms are almost certainly not exactly 1.0
            self.assertFalse(
                torch.allclose(raw_norms, expected_ones, atol=1e-3),
                "Unnormalized outputs unexpectedly have unit norm",
            )

    def test_feature_extraction(self) -> None:
        """
        Verify intermediate feature map extraction produces [B, 256, 14, 14]
        for standard input [B, 3, 224, 224].
        """
        model = ImageEncoder(embed_dim=256, feature_dim=256)
        model.eval()
        batch_size = 3
        x = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

        with torch.no_grad():
            features = model.extract_features(x)
            self.assertEqual(
                features.shape,
                (batch_size, 256, 14, 14),
                f"Expected feature map shape ({batch_size}, 256, 14, 14), got {features.shape}",
            )

    def test_gradient_backpropagation(self) -> None:
        """
        Verify backward pass computes non-NaN, non-Inf gradients for all
        convolutional, batchnorm, and linear projection parameters.
        """
        model = ImageEncoder(embed_dim=256)
        model.train()
        x = torch.rand(4, 3, 224, 224, dtype=torch.float32, requires_grad=True)

        embeddings = model(x, normalize=True)
        loss = embeddings.sum()
        loss.backward()

        # Check input gradient
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())

        # Check all named parameters have valid gradients
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

    def test_eval_and_train_modes(self) -> None:
        """
        Verify model properly toggles between train() and eval() modes.
        In eval mode, dropout is deactivated and BatchNorm uses running statistics.
        """
        model = ImageEncoder(embed_dim=256, dropout_p=0.5)

        # Train mode
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.projection_head[3].training)  # Dropout layer

        # Eval mode
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.projection_head[3].training)  # Dropout layer

        # In eval mode, multiple forward passes with same input must be deterministic
        x = torch.rand(2, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            self.assertTrue(
                torch.allclose(out1, out2, atol=1e-7),
                "Eval mode forward pass should be fully deterministic",
            )

    def test_invalid_inputs_raise_error(self) -> None:
        """
        Verify invalid input dimensions or channel counts raise ValueError.
        """
        model = ImageEncoder(in_channels=3, embed_dim=256)
        model.eval()

        # 3D tensor missing batch dimension [3, 224, 224]
        x_3d = torch.rand(3, 224, 224)
        with self.assertRaises(ValueError):
            model(x_3d)

        with self.assertRaises(ValueError):
            model.extract_features(x_3d)

        # 5D tensor [1, 3, 1, 224, 224]
        x_5d = torch.rand(1, 3, 1, 224, 224)
        with self.assertRaises(ValueError):
            model(x_5d)

        # Wrong channel count (e.g. grayscale 1 channel instead of 3)
        x_wrong_channels = torch.rand(2, 1, 224, 224)
        with self.assertRaises(ValueError):
            model(x_wrong_channels)

        with self.assertRaises(ValueError):
            model.extract_features(x_wrong_channels)


if __name__ == "__main__":
    unittest.main()
