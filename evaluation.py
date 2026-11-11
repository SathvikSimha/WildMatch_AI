"""
WildMatch AI - Member 4: Retrieval Evaluation & Metrics Suite (`evaluation.py`)

Implements quantitative retrieval evaluation metrics:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Both Species-Level Semantic Retrieval and Exact-Pair Retrieval paradigms
- Per-species breakdown, similarity margin analysis, and auxiliary classifier accuracy
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wildmatch_v3 import SPECIES_CLASSES
from retrieval import (
    WildMatchRetriever,
    calculate_cosine_similarity,
    resolve_project_path,
)


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    relevance_matrix: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Recall@K given a [N_queries, N_candidates] similarity matrix and
    boolean [N_queries, N_candidates] relevance mask.

    A query is counted as a hit at K if at least one relevant candidate appears
    within the Top-K highest-scoring candidates.
    """
    if similarity_matrix.shape != relevance_matrix.shape:
        raise ValueError(
            f"Shape mismatch: similarity {tuple(similarity_matrix.shape)} vs "
            f"relevance {tuple(relevance_matrix.shape)}"
        )
    if similarity_matrix.ndim != 2 or similarity_matrix.size(0) == 0:
        raise ValueError("similarity_matrix must be a non-empty 2D tensor.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    eff_k = min(k, similarity_matrix.size(1))
    topk_indices = torch.topk(similarity_matrix, k=eff_k, dim=1, largest=True, sorted=True).indices
    hits = torch.gather(relevance_matrix.bool(), dim=1, index=topk_indices).any(dim=1)
    return float(hits.double().mean().item())


def compute_mrr(
    similarity_matrix: torch.Tensor,
    relevance_matrix: torch.Tensor,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) over all queries.

    For each query, finds the 1-based rank of the highest-scoring relevant candidate
    and averages 1 / rank across all queries.
    """
    if similarity_matrix.shape != relevance_matrix.shape:
        raise ValueError(
            f"Shape mismatch: similarity {tuple(similarity_matrix.shape)} vs "
            f"relevance {tuple(relevance_matrix.shape)}"
        )
    if similarity_matrix.ndim != 2 or similarity_matrix.size(0) == 0:
        raise ValueError("similarity_matrix must be a non-empty 2D tensor.")

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    sorted_rel = torch.gather(relevance_matrix.bool(), dim=1, index=sorted_indices)

    reciprocal_ranks: List[float] = []
    for row in sorted_rel:
        pos = torch.nonzero(row, as_tuple=False)
        if pos.numel() == 0:
            reciprocal_ranks.append(0.0)
        else:
            first_rank = int(pos[0, 0].item()) + 1
            reciprocal_ranks.append(1.0 / first_rank)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    relevance_matrix: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute Recall@K (for each k in ks), MRR, and mean first-hit rank.
    """
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = compute_recall_at_k(similarity_matrix, relevance_matrix, k=k)
    metrics["mrr"] = compute_mrr(similarity_matrix, relevance_matrix)

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    sorted_rel = torch.gather(relevance_matrix.bool(), dim=1, index=sorted_indices)
    first_ranks: List[float] = []
    for row in sorted_rel:
        pos = torch.nonzero(row, as_tuple=False)
        if pos.numel() > 0:
            first_ranks.append(float(pos[0, 0].item() + 1))
    metrics["mean_first_rank"] = (
        sum(first_ranks) / len(first_ranks) if first_ranks else float("inf")
    )
    return metrics


def _build_relevance_matrices(
    species_list: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build [N, N] exact-pair identity relevance matrix and species-level relevance matrix.
    """
    n = len(species_list)
    exact_rel = torch.eye(n, dtype=torch.bool)
    species_rel = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if species_list[i] == species_list[j]:
                species_rel[i, j] = True
    return exact_rel, species_rel


def evaluate_retriever(
    retriever: WildMatchRetriever,
    split: str = "test",
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """
    Run full bidirectional retrieval and classification evaluation on a dataset split.

    Args:
        retriever: Initialized WildMatchRetriever (ideally indexed on 'all' or target split).
        split: Split to evaluate ('test', 'validation', 'train', or 'all').
        ks: Cutoffs for Recall@K (default: (1, 5, 10)).

    Returns:
        Structured dictionary containing aggregate metrics, per-species metrics,
        similarity margin statistics, and Member 3 reference comparison.
    """
    img_embs, txt_embs, records = retriever.get_paired_subset(split=split)
    n = len(records)
    if n == 0:
        raise RuntimeError(
            f"No paired image-text records available for split '{split}'. "
            "Ensure curated images exist under dataset/curated/."
        )

    species_labels = [rec["species"] for rec in records]
    exact_rel, species_rel = _build_relevance_matrices(species_labels)

    # [N_img, N_txt] cosine similarity matrix
    i2t_sim = calculate_cosine_similarity(img_embs, txt_embs)
    t2i_sim = i2t_sim.t()

    # Exact-pair retrieval
    i2t_exact = compute_retrieval_metrics(i2t_sim, exact_rel, ks=ks)
    t2i_exact = compute_retrieval_metrics(t2i_sim, exact_rel, ks=ks)

    # Species-level retrieval
    i2t_species = compute_retrieval_metrics(i2t_sim, species_rel, ks=ks)
    t2i_species = compute_retrieval_metrics(t2i_sim, species_rel, ks=ks)

    # Auxiliary classifier accuracy
    with torch.no_grad():
        _, img_preds, _ = retriever.heads.predict_image(img_embs.to(retriever.device))
        _, txt_preds, _ = retriever.heads.predict_text(txt_embs.to(retriever.device))

    img_cls_acc = sum(p == g for p, g in zip(img_preds, species_labels)) / n
    txt_cls_acc = sum(p == g for p, g in zip(txt_preds, species_labels)) / n

    # Similarity distribution statistics
    diag_sims = torch.diag(i2t_sim)
    same_sp_sims = i2t_sim[species_rel]
    diff_sp_sims = i2t_sim[~species_rel]

    mean_exact_sim = float(diag_sims.mean().item())
    mean_same_sp_sim = float(same_sp_sims.mean().item())
    mean_diff_sp_sim = float(diff_sp_sims.mean().item()) if diff_sp_sims.numel() > 0 else 0.0

    # Per-species breakdown
    per_species: Dict[str, Dict[str, Any]] = {}
    for sp in SPECIES_CLASSES:
        mask = torch.tensor([s == sp for s in species_labels], dtype=torch.bool)
        if not mask.any():
            continue
        sp_i2t_sp = compute_retrieval_metrics(i2t_sim[mask], species_rel[mask], ks=ks)
        sp_t2i_sp = compute_retrieval_metrics(t2i_sim[mask], species_rel[mask], ks=ks)
        sp_i2t_ex = compute_retrieval_metrics(i2t_sim[mask], exact_rel[mask], ks=ks)
        sp_t2i_ex = compute_retrieval_metrics(t2i_sim[mask], exact_rel[mask], ks=ks)
        per_species[sp] = {
            "count": int(mask.sum().item()),
            "t2i_species_r1": sp_t2i_sp["recall@1"],
            "t2i_species_r5": sp_t2i_sp["recall@5"],
            "t2i_species_mrr": sp_t2i_sp["mrr"],
            "i2t_species_r1": sp_i2t_sp["recall@1"],
            "i2t_species_r5": sp_i2t_sp["recall@5"],
            "i2t_species_mrr": sp_i2t_sp["mrr"],
            "t2i_exact_r1": sp_t2i_ex["recall@1"],
            "t2i_exact_r5": sp_t2i_ex["recall@5"],
            "t2i_exact_mrr": sp_t2i_ex["mrr"],
            "i2t_exact_r1": sp_i2t_ex["recall@1"],
            "i2t_exact_r5": sp_i2t_ex["recall@5"],
            "i2t_exact_mrr": sp_i2t_ex["mrr"],
        }

    summary: Dict[str, Any] = {
        "split": split,
        "num_samples": n,
        "species_counts": {
            sp: sum(1 for s in species_labels if s == sp) for sp in SPECIES_CLASSES
        },
        "metrics": {
            "t2i_species_r1": t2i_species["recall@1"],
            "t2i_species_r5": t2i_species["recall@5"],
            "t2i_species_r10": t2i_species["recall@10"],
            "t2i_species_mrr": t2i_species["mrr"],
            "i2t_species_r1": i2t_species["recall@1"],
            "i2t_species_r5": i2t_species["recall@5"],
            "i2t_species_r10": i2t_species["recall@10"],
            "i2t_species_mrr": i2t_species["mrr"],
            "species_r1_average": 0.5 * (t2i_species["recall@1"] + i2t_species["recall@1"]),
            "species_r5_average": 0.5 * (t2i_species["recall@5"] + i2t_species["recall@5"]),
            "species_r10_average": 0.5 * (t2i_species["recall@10"] + i2t_species["recall@10"]),
            "species_mrr_average": 0.5 * (t2i_species["mrr"] + i2t_species["mrr"]),
            "t2i_exact_r1": t2i_exact["recall@1"],
            "t2i_exact_r5": t2i_exact["recall@5"],
            "t2i_exact_r10": t2i_exact["recall@10"],
            "t2i_exact_mrr": t2i_exact["mrr"],
            "i2t_exact_r1": i2t_exact["recall@1"],
            "i2t_exact_r5": i2t_exact["recall@5"],
            "i2t_exact_r10": i2t_exact["recall@10"],
            "i2t_exact_mrr": i2t_exact["mrr"],
            "exact_r1_average": 0.5 * (t2i_exact["recall@1"] + i2t_exact["recall@1"]),
            "exact_r5_average": 0.5 * (t2i_exact["recall@5"] + i2t_exact["recall@5"]),
            "exact_r10_average": 0.5 * (t2i_exact["recall@10"] + i2t_exact["recall@10"]),
            "exact_mrr_average": 0.5 * (t2i_exact["mrr"] + i2t_exact["mrr"]),
            "image_classifier_accuracy": float(img_cls_acc),
            "text_classifier_accuracy": float(txt_cls_acc),
        },
        "similarity_stats": {
            "mean_exact_pair_similarity": mean_exact_sim,
            "mean_same_species_similarity": mean_same_sp_sim,
            "mean_diff_species_similarity": mean_diff_sp_sim,
            "species_separation_margin": mean_same_sp_sim - mean_diff_sp_sim,
        },
        "per_species": per_species,
    }

    # Include comparison with Member 3's recorded test metrics if evaluating test split
    ref_path = resolve_project_path("models/wildmatch_v3_metrics.json")
    if split.lower() == "test" and ref_path.is_file():
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = json.load(f)
        summary["member3_reference_test_metrics"] = ref

    return summary


def format_evaluation_report(eval_result: Dict[str, Any]) -> str:
    """Format an evaluation dictionary as a readable terminal / markdown report."""
    m = eval_result["metrics"]
    s = eval_result["similarity_stats"]
    lines = [
        "=" * 80,
        f"WILDMATCH AI — MEMBER 4 RETRIEVAL EVALUATION (Split: {eval_result['split'].upper()}, N={eval_result['num_samples']})",
        "=" * 80,
        "",
        "1. Species-Level Semantic Retrieval (Primary Objective):",
        f"   Text-to-Image : Recall@1 = {m['t2i_species_r1']*100:6.2f}% | Recall@5 = {m['t2i_species_r5']*100:6.2f}% | Recall@10 = {m['t2i_species_r10']*100:6.2f}% | MRR = {m['t2i_species_mrr']:.4f}",
        f"   Image-to-Text : Recall@1 = {m['i2t_species_r1']*100:6.2f}% | Recall@5 = {m['i2t_species_r5']*100:6.2f}% | Recall@10 = {m['i2t_species_r10']*100:6.2f}% | MRR = {m['i2t_species_mrr']:.4f}",
        f"   Bimodal Mean  : Recall@1 = {m['species_r1_average']*100:6.2f}% | Recall@5 = {m['species_r5_average']*100:6.2f}% | Recall@10 = {m['species_r10_average']*100:6.2f}% | MRR = {m['species_mrr_average']:.4f}",
        "",
        "2. Exact Image-Caption Pair Retrieval (Secondary Objective):",
        f"   Text-to-Image : Recall@1 = {m['t2i_exact_r1']*100:6.2f}% | Recall@5 = {m['t2i_exact_r5']*100:6.2f}% | Recall@10 = {m['t2i_exact_r10']*100:6.2f}% | MRR = {m['t2i_exact_mrr']:.4f}",
        f"   Image-to-Text : Recall@1 = {m['i2t_exact_r1']*100:6.2f}% | Recall@5 = {m['i2t_exact_r5']*100:6.2f}% | Recall@10 = {m['i2t_exact_r10']*100:6.2f}% | MRR = {m['i2t_exact_mrr']:.4f}",
        f"   Bimodal Mean  : Recall@1 = {m['exact_r1_average']*100:6.2f}% | Recall@5 = {m['exact_r5_average']*100:6.2f}% | Recall@10 = {m['exact_r10_average']*100:6.2f}% | MRR = {m['exact_mrr_average']:.4f}",
        "",
        "3. Auxiliary Species Classifier Accuracy & Embedding Geometry:",
        f"   Image Classifier Acc     : {m['image_classifier_accuracy']*100:6.2f}%",
        f"   Text Classifier Acc      : {m['text_classifier_accuracy']*100:6.2f}%",
        f"   Mean Exact-Pair Cosine   : {s['mean_exact_pair_similarity']:+.4f}",
        f"   Mean Same-Species Cosine : {s['mean_same_species_similarity']:+.4f}",
        f"   Mean Diff-Species Cosine : {s['mean_diff_species_similarity']:+.4f}",
        f"   Species Separation Margin: {s['species_separation_margin']:+.4f}",
        "",
        "4. Per-Species Breakdown (Species-Level Retrieval):",
    ]
    for sp, sp_m in eval_result["per_species"].items():
        lines.append(
            f"   {sp:<15} (N={sp_m['count']:>2}) | "
            f"T2I R@1={sp_m['t2i_species_r1']*100:5.1f}% MRR={sp_m['t2i_species_mrr']:.3f} | "
            f"I2T R@1={sp_m['i2t_species_r1']*100:5.1f}% R@5={sp_m['i2t_species_r5']*100:5.1f}% MRR={sp_m['i2t_species_mrr']:.3f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: Union[str, Path] = "docs/member4_evaluation_results.json",
) -> Path:
    """Persist evaluation metrics to JSON."""
    out = resolve_project_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WildMatch AI Member 4 Retrieval Evaluation Suite"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test", "all"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="docs/member4_evaluation_results.json",
        help="Path to save JSON evaluation report.",
    )
    args = parser.parse_args()

    retriever = WildMatchRetriever(split=args.split)
    eval_result = evaluate_retriever(retriever, split=args.split)
    print(format_evaluation_report(eval_result))

    if args.save_json and args.save_json.strip().lower() not in ("none", ""):
        saved = save_evaluation_results(eval_result, args.save_json)
        print(f"[OK] Saved evaluation results to {saved}")


if __name__ == "__main__":
    main()