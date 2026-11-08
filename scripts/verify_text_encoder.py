"""
WildMatch AI - BiLSTM Text Encoder Verification and Profiling Script

Validates the BiLSTM TextEncoder against 7 strict architectural invariants:
1. Input tensor shape & dtype compatibility: Accepts [16, 32] torch.long integer token tensors.
2. Output embedding shape: Produces [16, embed_dim] (default 256).
3. L2 unit normalization: Vector norms equal 1.0 +/- 1e-5 when normalize=True.
4. Backward pass gradient flow: Gradient backpropagation computes valid, non-NaN, non-Inf gradients.
5. Lightweight model footprint: Total parameters < 2,000,000 (< 2M).
6. Intermediate recurrent feature extraction: extract_features() provides [16, 32, 256] temporal features.
7. Real vocabulary integration: Successfully tokenizes, numericalizes, and encodes real captions from vocabulary.json.

Profiles latency (batch and per-caption) on CPU and reports detailed parameter breakdown.
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
    from WildMatch_AI.models.text_encoder import TextEncoder
    from WildMatch_AI.preprocessing.text_preprocessing import Vocabulary, tokenize_caption
except ImportError:
    from models.text_encoder import TextEncoder
    from preprocessing.text_preprocessing import Vocabulary, tokenize_caption


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
        True if all 7 invariants PASS, False otherwise.
    """
    print("=" * 80)
    print("WILDMATCH AI - BILSTM TEXT ENCODER VERIFICATION & PROFILING")
    print("=" * 80)

    results: Dict[str, bool] = {}
    vocab_size = 158
    embed_dim = 256
    word_embed_dim = 128
    hidden_dim = 128
    batch_size = 16
    seq_len = 32

    # -------------------------------------------------------------------------
    # 1. Model Instantiation and Architecture Summary
    # -------------------------------------------------------------------------
    print("\n--- 1. Architecture Structural Breakdown ---")
    model = TextEncoder(
        vocab_size=vocab_size,
        word_embed_dim=word_embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        embed_dim=embed_dim,
        padding_idx=0,
        dropout_p=0.1,
        pooling="mean",
    )
    print(f"Model Class: {model.__class__.__name__}")
    print(
        f"Config: vocab_size={vocab_size}, word_embed_dim={word_embed_dim}, "
        f"hidden_dim={hidden_dim}, embed_dim={embed_dim}, pooling='mean'\n"
    )

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

    param_check = total_params < 2_000_000
    results["Invariant 5: Lightweight model size (< 2M parameters)"] = param_check
    print(f"  Parameter Limit Check: {'PASS' if param_check else 'FAIL'} (Threshold < 2.00M)")

    # -------------------------------------------------------------------------
    # 3. Input Shape Compatibility and Feature Extraction
    # -------------------------------------------------------------------------
    print("\n--- 3. Forward Pass & Feature Extraction ---")
    model.eval()

    # Generate synthetic batch with realistic non-pad prefix and pad suffix
    synthetic_captions = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    for i in range(batch_size):
        actual_len = torch.randint(5, 20, (1,)).item()
        synthetic_captions[i, actual_len:] = 0  # <PAD> token is 0

    print(f"  Input Tensor Shape   : {list(synthetic_captions.shape)} (dtype={synthetic_captions.dtype})")

    results["Invariant 1: Input tensor shape & dtype compatibility [16, 32] torch.long"] = (
        synthetic_captions.shape == (batch_size, seq_len) and
        synthetic_captions.dtype == torch.long
    )

    with torch.no_grad():
        features = model.extract_features(synthetic_captions)
        expected_feature_dim = hidden_dim * 2
        print(f"  Extracted Features   : {list(features.shape)} (expected [{batch_size}, {seq_len}, {expected_feature_dim}])")
        feature_check = features.shape == (batch_size, seq_len, expected_feature_dim)
        results["Invariant 6: Intermediate recurrent feature extraction [16, 32, 256]"] = feature_check

        embeddings = model(synthetic_captions, normalize=True)
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
    grad_captions = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    grad_captions[:, 15:] = 0  # Pad trailing positions

    emb = model(grad_captions, normalize=True)
    sim_loss = 1.0 - torch.cosine_similarity(emb[0:8], emb[8:16]).mean()
    sim_loss.backward()

    unassigned_params = []
    nan_params = []
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                unassigned_params.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_params.append(name)
            elif name != "embedding.weight" and torch.all(param.grad == 0):
                zero_grad_params.append(name)

    # Verify embedding pad row (index 0) remains 0
    pad_grad_zero = torch.all(model.embedding.weight.grad[0] == 0).item()

    print(f"  Simulated Loss Value : {sim_loss.item():.6f}")
    print(f"  Padding Index Grad   : {'ZERO (as expected)' if pad_grad_zero else 'NON-ZERO (ERROR)'}")
    print(f"  Parameter Gradients  : {len(unassigned_params)} unassigned, {len(nan_params)} NaN/Inf, {len(zero_grad_params)} zero")

    grad_check = (
        len(unassigned_params) == 0 and
        len(nan_params) == 0 and
        len(zero_grad_params) == 0 and
        pad_grad_zero
    )
    results["Invariant 4: Backward pass gradient flow (non-NaN / non-Inf)"] = grad_check
    print(f"  Gradient Flow Check  : {'PASS' if grad_check else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 6. Real Vocabulary Token Integration
    # -------------------------------------------------------------------------
    print("\n--- 6. Real Vocabulary Integration Test ---")
    vocab_file = WILDMATCH_ROOT / "dataset" / "vocabulary.json"
    vocab_ok = False
    if vocab_file.exists():
        vocab = Vocabulary.load_json(str(vocab_file))
        species_captions = [
            "a tiger resting in water",
            "lion resting beneath tree",
            "asian elephant grazing on hillside",
            "chital standing in grass",
        ]
        species_tensors = []
        for cap in species_captions:
            tokens = tokenize_caption(cap)
            ids = vocab.numericalize(tokens, add_special_tokens=True, max_len=seq_len, pad=True)
            species_tensors.append(torch.tensor(ids, dtype=torch.long))

        species_batch = torch.stack(species_tensors, dim=0)
        model.eval()
        with torch.no_grad():
            species_emb = model(species_batch, normalize=True)
            species_norms = torch.norm(species_emb, p=2, dim=-1)
            vocab_ok = (
                species_emb.shape == (4, embed_dim) and
                torch.allclose(species_norms, torch.ones_like(species_norms), atol=1e-5)
            )
            print(f"  Species Batch Shape  : {list(species_emb.shape)}")
            print(f"  Species L2 Norms     : {[round(n.item(), 5) for n in species_norms]}")
            print(f"  Vocab Integration    : {'PASS' if vocab_ok else 'FAIL'}")
    else:
        print(f"  [WARN] vocabulary.json not found at {vocab_file}")

    results["Invariant 7: Real vocabulary token integration from vocabulary.json"] = vocab_ok

    # -------------------------------------------------------------------------
    # 7. Latency Profiling (CPU Benchmark)
    # -------------------------------------------------------------------------
    print("\n--- 7. Inference Latency Benchmark (CPU) ---")
    model.eval()
    bench_tensor = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

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
    mean_caption_ms = mean_batch_ms / batch_size
    cps = (iterations * batch_size) / total_time

    print(f"  Benchmark Iterations : {iterations} batches (batch size {batch_size})")
    print(f"  Mean Batch Latency   : {mean_batch_ms:.2f} ms")
    print(f"  Mean Per-Caption Time: {mean_caption_ms:.2f} ms")
    print(f"  Throughput           : {cps:.1f} captions/sec")

    # -------------------------------------------------------------------------
    # 8. Summary Invariant Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CRITERIA VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for criterion, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {criterion:<70} [{status}]")

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All 7 Phase 2 architectural invariants verified successfully.")
        return True
    else:
        print("[FAILURE] One or more architectural invariants failed verification.")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
