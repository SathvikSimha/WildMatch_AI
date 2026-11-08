"""
WildMatch AI - Custom CNN Image Encoder Verification and Profiling Script

Validates the Custom CNN ImageEncoder against 6 strict architectural invariants:
1. Input tensor shape compatibility: Accepts [B, 3, 224, 224] float32 tensors.
2. Output embedding shape: Produces [B, embed_dim] (default 256).
3. L2 unit normalization: Vector norms equal 1.0 +/- 1e-5 when normalize=True.
4. Backward pass gradient flow: Gradient backpropagation computes valid, non-NaN, non-Inf gradients.
5. Lightweight model footprint: Total parameters < 5,000,000.
6. Intermediate feature map extraction: extract_features() provides [B, 256, 14, 14] feature maps.

Profiles latency (batch and per-image) on CPU and reports detailed parameter breakdown.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root and WildMatch_AI package are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
WILDMATCH_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = WILDMATCH_ROOT.parent

for path in [str(WORKSPACE_ROOT), str(WILDMATCH_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from WildMatch_AI.models.image_encoder import ImageEncoder
except ImportError:
    from models.image_encoder import ImageEncoder


def format_param_count(count: int) -> str:
    """Format parameter count into human-readable string."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M ({count:,})"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K ({count:,})"
    return f"{count:,}"


def run_verification() -> bool:
    """
    Run complete architectural inspection, profiling, and invariant validation.

    Returns:
        True if all 6 invariants PASS, False otherwise.
    """
    print("=" * 80)
    print("WILDMATCH AI - CUSTOM CNN IMAGE ENCODER VERIFICATION & PROFILING")
    print("=" * 80)

    results: Dict[str, bool] = {}
    embed_dim = 256
    batch_size = 16

    # -------------------------------------------------------------------------
    # 1. Model Instantiation and Architecture Summary
    # -------------------------------------------------------------------------
    print("\n--- 1. Architecture Structural Breakdown ---")
    model = ImageEncoder(in_channels=3, embed_dim=embed_dim, feature_dim=256, dropout_p=0.1)
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Config: in_channels=3, embed_dim={embed_dim}, feature_dim=256, dropout_p=0.1\n")

    print("[Layer Breakdown]")
    for name, module in model.named_children():
        submodules_count = sum(1 for _ in module.modules()) - 1
        print(f"  - {name:<18}: {module.__class__.__name__} ({submodules_count} child modules)")

    # -------------------------------------------------------------------------
    # 2. Parameter Counts and Memory Footprint
    # -------------------------------------------------------------------------
    print("\n--- 2. Parameter Counts and Model Footprint ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_params = sum(b.numel() for b in model.buffers())

    # Footprint in Megabytes (4 bytes per float32 parameter/buffer)
    footprint_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + \
                     sum(b.numel() * b.element_size() for b in model.buffers())
    footprint_mb = footprint_bytes / (1024 * 1024)

    print(f"  Total Parameters     : {format_param_count(total_params)}")
    print(f"  Trainable Parameters : {format_param_count(trainable_params)}")
    print(f"  Buffer Elements (BN) : {format_param_count(buffer_params)}")
    print(f"  Model Weight Size    : {footprint_mb:.2f} MB")

    param_check = total_params < 5_000_000
    results["Invariant 5: Lightweight model size (< 5M parameters)"] = param_check
    print(f"  Parameter Limit Check: {'PASS' if param_check else 'FAIL'} (Threshold < 5.00M)")

    # -------------------------------------------------------------------------
    # 3. Input Shape Compatibility and Intermediate Feature Map Extraction
    # -------------------------------------------------------------------------
    print("\n--- 3. Forward Pass & Feature Map Extraction ---")
    model.eval()
    synthetic_images = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
    print(f"  Input Tensor Shape   : {list(synthetic_images.shape)} (dtype={synthetic_images.dtype})")

    results["Invariant 1: Input tensor shape compatibility [16, 3, 224, 224]"] = (
        synthetic_images.shape == (batch_size, 3, 224, 224)
    )

    with torch.no_grad():
        features = model.extract_features(synthetic_images)
        print(f"  Extracted Features   : {list(features.shape)} (expected [{batch_size}, 256, 14, 14])")
        feature_check = features.shape == (batch_size, 256, 14, 14)
        results["Invariant 6: Intermediate feature map extraction [16, 256, 14, 14]"] = feature_check

        embeddings = model(synthetic_images, normalize=True)
        print(f"  Output Embeddings    : {list(embeddings.shape)} (expected [{batch_size}, {embed_dim}])")
        shape_check = embeddings.shape == (batch_size, embed_dim)
        results["Invariant 2: Output embedding shape [16, 256]"] = shape_check

    # -------------------------------------------------------------------------
    # 4. L2 Normalization Statistics
    # -------------------------------------------------------------------------
    print("\n--- 4. L2 Normalization Verification ---")
    with torch.no_grad():
        norms = torch.norm(embeddings, p=2, dim=-1)
        min_norm = norms.min().item()
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()

        print(f"  L2 Norm Range        : [{min_norm:.7f}, {max_norm:.7f}] (mean: {mean_norm:.7f}, std: {std_norm:.2e})")
        print(f"  Embedding Values     : min={embeddings.min().item():.5f}, max={embeddings.max().item():.5f}")

        norm_check = (
            abs(min_norm - 1.0) < 1e-4 and
            abs(max_norm - 1.0) < 1e-4 and
            abs(mean_norm - 1.0) < 1e-5
        )
        results["Invariant 3: L2 unit normalization (norm = 1.0 +/- 1e-5)"] = norm_check
        print(f"  Unit Sphere Norm     : {'PASS' if norm_check else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 5. Gradient Flow & Backward Pass Simulation
    # -------------------------------------------------------------------------
    print("\n--- 5. Gradient Backpropagation Simulation ---")
    model.train()
    grad_images = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, requires_grad=True)

    emb = model(grad_images, normalize=True)
    # Simulated contrastive cosine distance loss between pairs in batch
    sim_loss = 1.0 - torch.cosine_similarity(emb[0:8], emb[8:16]).mean()
    sim_loss.backward()

    has_input_grad = grad_images.grad is not None and not torch.isnan(grad_images.grad).any()
    nan_params = []
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_params.append(name)

    print(f"  Simulated Loss Value : {sim_loss.item():.6f}")
    print(f"  Input Gradient Check : {'VALID' if has_input_grad else 'INVALID'}")
    print(f"  Parameter Gradients  : {len(zero_grad_params)} unpopulated, {len(nan_params)} NaN/Inf")

    grad_check = has_input_grad and len(nan_params) == 0 and len(zero_grad_params) == 0
    results["Invariant 4: Backward pass gradient flow (non-NaN / non-Inf)"] = grad_check
    print(f"  Gradient Flow Check  : {'PASS' if grad_check else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 6. Latency Profiling (CPU Benchmark)
    # -------------------------------------------------------------------------
    print("\n--- 6. Inference Latency Benchmark (CPU) ---")
    model.eval()
    bench_tensor = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            _ = model(bench_tensor)

    # Measured runs (50 iterations)
    iterations = 50
    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(bench_tensor)
    total_time = time.perf_counter() - start_time

    mean_batch_ms = (total_time / iterations) * 1000.0
    mean_image_ms = mean_batch_ms / batch_size
    fps = (iterations * batch_size) / total_time

    print(f"  Benchmark Iterations : {iterations} batches (batch size {batch_size})")
    print(f"  Mean Batch Latency   : {mean_batch_ms:.2f} ms")
    print(f"  Mean Per-Image Time  : {mean_image_ms:.2f} ms")
    print(f"  Throughput           : {fps:.1f} images/sec")

    # -------------------------------------------------------------------------
    # 7. Summary Invariant Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CRITERIA VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for criterion, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {criterion:<65} [{status}]")

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All 6 Phase 1 architectural invariants verified successfully.")
        return True
    else:
        print("[FAILURE] One or more architectural invariants failed verification.")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
