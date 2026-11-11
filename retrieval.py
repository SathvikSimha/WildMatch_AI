"""
WildMatch AI - Member 4: Similarity Matching & Multimodal Retrieval Engine (`retrieval.py`)

Implements cosine similarity calculation, offline gallery vector indexing,
bidirectional semantic retrieval (Text-to-Image and Image-to-Text), Top-K
ranking, and pairwise cross-modal scoring using Member 3's trained Version 3
Dual Encoder and Auxiliary Species Heads.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch
import torch.nn.functional as F

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference import (
    load_model,
    load_species_heads,
    load_vocabulary,
    preprocess_image,
    tokenize_caption,
    embed_image,
    embed_text,
    compute_similarity,
)
from models.wildmatch_v3 import (
    WildMatchV3DualEncoder,
    WildMatchV3SpeciesHeads,
    SPECIES_CLASSES,
)


SPLIT_FILE_MAP: Dict[str, str] = {
    "train": "train.csv",
    "validation": "validation.csv",
    "val": "validation.csv",
    "test": "test.csv",
}


def calculate_cosine_similarity(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    ensure_normalized: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between query and candidate embeddings.

    Args:
        query_embeddings: Tensor of shape [D] or [N_q, D].
        candidate_embeddings: Tensor of shape [D] or [N_c, D].
        ensure_normalized: L2-normalize vectors prior to dot product (default: True).

    Returns:
        Cosine similarity matrix of shape [N_q, N_c] with values in [-1.0, 1.0].
    """
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.unsqueeze(0)
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.unsqueeze(0)

    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D embeddings, got shapes "
            f"{tuple(query_embeddings.shape)} and {tuple(candidate_embeddings.shape)}"
        )

    if ensure_normalized:
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)

    sim = compute_similarity(query_embeddings, candidate_embeddings)
    return sim.clamp(min=-1.0, max=1.0)


def resolve_project_path(path_value: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root if it is not already absolute."""
    p = Path(path_value)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


def load_split_dataframe(
    splits_dir: Union[str, Path] = "dataset/splits",
    split: str = "all",
) -> pd.DataFrame:
    """
    Load one split ('train', 'validation', 'test') or all splits combined ('all').
    """
    splits_path = resolve_project_path(splits_dir)
    split_key = split.lower().strip()

    if split_key == "all":
        frames: List[pd.DataFrame] = []
        for split_name in ("train", "validation", "test"):
            csv_path = splits_path / SPLIT_FILE_MAP[split_name]
            if not csv_path.is_file():
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            df_part = pd.read_csv(csv_path)
            df_part["split"] = split_name
            frames.append(df_part)
        combined = pd.concat(frames, ignore_index=True)
        return combined

    if split_key not in SPLIT_FILE_MAP:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of: 'train', 'validation', 'test', 'all'."
        )

    csv_path = splits_path / SPLIT_FILE_MAP[split_key]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["split"] = "validation" if split_key == "val" else split_key
    return df


class WildMatchRetriever:
    """
    End-to-end multimodal similarity matching and retrieval engine for WildMatch AI.

    Loads Member 3's trained ResNet-18 + BiLSTM Dual Encoder and auxiliary species
    classification heads, precomputes normalized 256-d gallery embeddings, and serves
    fast Top-K Text-to-Image, Image-to-Text, and custom Image-Text pair scoring.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/wildmatch_v3_dual_encoder.pth",
        heads_path: Union[str, Path] = "models/wildmatch_v3_species_heads.pth",
        vocab_path: Union[str, Path] = "models/wildmatch_v3_vocabulary.json",
        splits_dir: Union[str, Path] = "dataset/splits",
        split: str = "all",
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
        auto_index: bool = True,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.heads_path = resolve_project_path(heads_path)
        self.vocab_path = resolve_project_path(vocab_path)
        self.splits_dir = resolve_project_path(splits_dir)
        self.batch_size = batch_size
        self.split = split

        self.vocab: Dict[str, Any] = load_vocabulary(self.vocab_path)
        self.model: WildMatchV3DualEncoder = load_model(
            self.checkpoint_path, device=self.device
        )
        self.heads: WildMatchV3SpeciesHeads = load_species_heads(
            self.heads_path, device=self.device
        )

        # Gallery index state
        self.records: List[Dict[str, Any]] = []
        self.image_embeddings: Optional[torch.Tensor] = None  # [N_img, 256]
        self.image_record_indices: List[int] = []
        self.text_embeddings: Optional[torch.Tensor] = None   # [N_txt, 256]
        self.missing_image_paths: List[str] = []

        if auto_index:
            self.build_index(split=split)

    def build_index(
        self,
        split: str = "all",
        df_override: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Precompute and cache normalized 256-d embeddings for gallery images and captions.

        Args:
            split: Dataset split to index ('train', 'validation', 'test', or 'all').
            df_override: Optional pre-loaded DataFrame instead of reading split CSVs.

        Returns:
            Summary dictionary describing indexed image and caption counts.
        """
        self.split = split
        df = df_override.copy() if df_override is not None else load_split_dataframe(
            splits_dir=self.splits_dir, split=split
        )

        records: List[Dict[str, Any]] = []
        captions: List[str] = []
        img_tensors: List[torch.Tensor] = []
        img_record_indices: List[int] = []
        missing_paths: List[str] = []

        for idx, row in df.iterrows():
            raw_img_path = str(row["image_path"])
            resolved_img_path = resolve_project_path(raw_img_path)
            caption_text = str(
                row.get("final_caption")
                or row.get("cleaned_caption")
                or row.get("blip_caption")
                or ""
            )

            rec_idx = len(records)
            record = {
                "index": rec_idx,
                "image_id": str(row["image_id"]),
                "photo_id": int(row["photo_id"]),
                "species": str(row["species"]),
                "image_path": str(resolved_img_path),
                "relative_image_path": raw_img_path,
                "caption": caption_text,
                "cleaned_caption": str(row.get("cleaned_caption") or caption_text),
                "split": str(row.get("split", split)),
                "has_image_file": resolved_img_path.is_file(),
            }
            records.append(record)
            captions.append(caption_text)

            if resolved_img_path.is_file():
                img_tensor = preprocess_image(resolved_img_path, device="cpu")
                img_tensors.append(img_tensor)
                img_record_indices.append(rec_idx)
            else:
                missing_paths.append(raw_img_path)

        # Encode all captions in batches
        text_emb_chunks: List[torch.Tensor] = []
        for start in range(0, len(captions), self.batch_size):
            batch_caps = captions[start : start + self.batch_size]
            emb_chunk = embed_text(
                self.model,
                batch_caps,
                vocab_or_path=self.vocab,
                normalize=True,
                device=self.device,
            )
            text_emb_chunks.append(emb_chunk.cpu())

        self.text_embeddings = (
            torch.cat(text_emb_chunks, dim=0)
            if text_emb_chunks
            else torch.empty((0, self.model.embed_dim), dtype=torch.float32)
        )

        # Encode all available images in batches
        img_emb_chunks: List[torch.Tensor] = []
        for start in range(0, len(img_tensors), self.batch_size):
            batch_imgs = torch.cat(img_tensors[start : start + self.batch_size], dim=0).to(
                self.device
            )
            emb_chunk = embed_image(self.model, batch_imgs, normalize=True)
            img_emb_chunks.append(emb_chunk.cpu())

        self.image_embeddings = (
            torch.cat(img_emb_chunks, dim=0)
            if img_emb_chunks
            else torch.empty((0, self.model.embed_dim), dtype=torch.float32)
        )

        # Cache auxiliary species predictions per record for rich UI display
        with torch.no_grad():
            if self.text_embeddings.size(0) > 0:
                _, txt_labels, txt_probs = self.heads.predict_text(
                    self.text_embeddings.to(self.device)
                )
                txt_probs_cpu = txt_probs.cpu()
                for i, rec in enumerate(records):
                    rec["text_pred_species"] = txt_labels[i]
                    rec["text_pred_confidence"] = float(txt_probs_cpu[i].max().item())

            if self.image_embeddings.size(0) > 0:
                _, img_labels, img_probs = self.heads.predict_image(
                    self.image_embeddings.to(self.device)
                )
                img_probs_cpu = img_probs.cpu()
                for pos, rec_idx in enumerate(img_record_indices):
                    records[rec_idx]["image_pred_species"] = img_labels[pos]
                    records[rec_idx]["image_pred_confidence"] = float(
                        img_probs_cpu[pos].max().item()
                    )

        self.records = records
        self.image_record_indices = img_record_indices
        self.missing_image_paths = missing_paths

        return {
            "split": split,
            "total_records": len(records),
            "indexed_images": len(img_record_indices),
            "indexed_captions": len(captions),
            "missing_images": len(missing_paths),
        }

    def _filter_indices(
        self,
        candidate_record_indices: Sequence[int],
        species_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ) -> List[int]:
        """Return positional indices within candidate_record_indices matching optional filters."""
        selected: List[int] = []
        sp_norm = species_filter.strip().lower() if species_filter and species_filter != "All" else None
        split_norm = split_filter.strip().lower() if split_filter and split_filter != "all" else None

        for pos, rec_idx in enumerate(candidate_record_indices):
            rec = self.records[rec_idx]
            if sp_norm is not None and rec["species"].lower() != sp_norm:
                continue
            if split_norm is not None and rec["split"].lower() != split_norm:
                continue
            selected.append(pos)
        return selected

    def _predict_text_species(self, text_emb: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        with torch.no_grad():
            _, labels, probs = self.heads.predict_text(text_emb.to(self.device))
        prob_vec = probs[0].cpu().tolist()
        prob_map = {sp: float(p) for sp, p in zip(self.heads.species, prob_vec)}
        return labels[0], float(max(prob_vec)), prob_map

    def _predict_image_species(self, img_emb: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        with torch.no_grad():
            _, labels, probs = self.heads.predict_image(img_emb.to(self.device))
        prob_vec = probs[0].cpu().tolist()
        prob_map = {sp: float(p) for sp, p in zip(self.heads.species, prob_vec)}
        return labels[0], float(max(prob_vec)), prob_map

    def retrieve_images_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        species_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Text-to-Image semantic retrieval: encode a natural language query and return
        Top-K matching wildlife images ranked by cosine similarity.
        """
        if not query_text or not str(query_text).strip():
            raise ValueError("Query text cannot be empty.")
        if self.image_embeddings is None or self.image_embeddings.size(0) == 0:
            raise RuntimeError(
                "No gallery image embeddings are indexed. Ensure curated images exist under dataset/curated/."
            )

        query_emb = embed_text(
            self.model,
            query_text,
            vocab_or_path=self.vocab,
            normalize=True,
            device=self.device,
        ).cpu()

        pred_species, pred_conf, prob_map = self._predict_text_species(query_emb)

        # Check vocabulary coverage for user feedback
        tokens = tokenize_caption(query_text, vocab_or_path=self.vocab)
        token_list = tokens[0].tolist()
        word_tokens = [t for t in token_list if t not in (0, 2, 3)]
        unk_count = sum(1 for t in word_tokens if t == 1)

        candidate_positions = self._filter_indices(
            self.image_record_indices,
            species_filter=species_filter,
            split_filter=split_filter,
        )
        if not candidate_positions:
            return {
                "query": query_text,
                "predicted_species": pred_species,
                "predicted_confidence": pred_conf,
                "species_probabilities": prob_map,
                "unk_token_count": unk_count,
                "total_tokens": len(word_tokens),
                "results": [],
            }

        sub_embs = self.image_embeddings[candidate_positions]
        sims = calculate_cosine_similarity(query_emb, sub_embs).squeeze(0)  # [N_cand]

        k = min(max(1, int(top_k)), sims.numel())
        top_scores, top_pos_indices = torch.topk(sims, k=k, largest=True, sorted=True)

        results: List[Dict[str, Any]] = []
        for rank, (score, local_pos) in enumerate(
            zip(top_scores.tolist(), top_pos_indices.tolist()), start=1
        ):
            gallery_pos = candidate_positions[local_pos]
            rec_idx = self.image_record_indices[gallery_pos]
            rec = self.records[rec_idx]
            results.append(
                {
                    "rank": rank,
                    "similarity": float(score),
                    "image_id": rec["image_id"],
                    "photo_id": rec["photo_id"],
                    "species": rec["species"],
                    "predicted_species": rec.get("image_pred_species", rec["species"]),
                    "predicted_confidence": rec.get("image_pred_confidence", 0.0),
                    "image_path": rec["image_path"],
                    "relative_image_path": rec["relative_image_path"],
                    "caption": rec["caption"],
                    "split": rec["split"],
                }
            )

        return {
            "query": query_text,
            "predicted_species": pred_species,
            "predicted_confidence": pred_conf,
            "species_probabilities": prob_map,
            "unk_token_count": unk_count,
            "total_tokens": len(word_tokens),
            "results": results,
        }

    def retrieve_texts_by_image(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 5,
        species_filter: Optional[str] = None,
        split_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Image-to-Text semantic retrieval: encode a wildlife image and return
        Top-K matching observational captions ranked by cosine similarity.
        """
        if self.text_embeddings is None or self.text_embeddings.size(0) == 0:
            raise RuntimeError("No gallery caption embeddings are indexed.")

        img_tensor = preprocess_image(image_input, device=self.device)
        query_emb = embed_image(self.model, img_tensor, normalize=True).cpu()

        pred_species, pred_conf, prob_map = self._predict_image_species(query_emb)

        all_indices = list(range(len(self.records)))
        candidate_positions = self._filter_indices(
            all_indices,
            species_filter=species_filter,
            split_filter=split_filter,
        )
        if not candidate_positions:
            return {
                "predicted_species": pred_species,
                "predicted_confidence": pred_conf,
                "species_probabilities": prob_map,
                "results": [],
            }

        sub_embs = self.text_embeddings[candidate_positions]
        sims = calculate_cosine_similarity(query_emb, sub_embs).squeeze(0)  # [N_cand]

        k = min(max(1, int(top_k)), sims.numel())
        top_scores, top_pos_indices = torch.topk(sims, k=k, largest=True, sorted=True)

        results: List[Dict[str, Any]] = []
        for rank, (score, local_pos) in enumerate(
            zip(top_scores.tolist(), top_pos_indices.tolist()), start=1
        ):
            rec_idx = candidate_positions[local_pos]
            rec = self.records[rec_idx]
            results.append(
                {
                    "rank": rank,
                    "similarity": float(score),
                    "image_id": rec["image_id"],
                    "photo_id": rec["photo_id"],
                    "species": rec["species"],
                    "predicted_species": rec.get("text_pred_species", rec["species"]),
                    "predicted_confidence": rec.get("text_pred_confidence", 0.0),
                    "caption": rec["caption"],
                    "image_path": rec["image_path"],
                    "relative_image_path": rec["relative_image_path"],
                    "split": rec["split"],
                }
            )

        return {
            "predicted_species": pred_species,
            "predicted_confidence": pred_conf,
            "species_probabilities": prob_map,
            "results": results,
        }

    def score_image_text_pair(
        self,
        image_input: Union[str, Path, Image.Image],
        caption: str,
    ) -> Dict[str, Any]:
        """
        Compute cross-modal cosine similarity and auxiliary species predictions for
        an arbitrary image-caption pair.
        """
        if not caption or not str(caption).strip():
            raise ValueError("Caption text cannot be empty.")

        img_tensor = preprocess_image(image_input, device=self.device)
        img_emb = embed_image(self.model, img_tensor, normalize=True).cpu()
        txt_emb = embed_text(
            self.model,
            caption,
            vocab_or_path=self.vocab,
            normalize=True,
            device=self.device,
        ).cpu()

        sim = float(calculate_cosine_similarity(img_emb, txt_emb).item())
        scale = float(self.model.logit_scale.exp().clamp(max=self.model.max_temperature).item())
        img_sp, img_conf, img_probs = self._predict_image_species(img_emb)
        txt_sp, txt_conf, txt_probs = self._predict_text_species(txt_emb)

        return {
            "cosine_similarity": sim,
            "scaled_logit": sim * scale,
            "image_predicted_species": img_sp,
            "image_confidence": img_conf,
            "image_species_probabilities": img_probs,
            "text_predicted_species": txt_sp,
            "text_confidence": txt_conf,
            "text_species_probabilities": txt_probs,
            "species_agreement": img_sp == txt_sp,
        }

    def get_paired_subset(
        self, split: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Return aligned (image_embeddings, text_embeddings, records) for evaluation
        where both image and caption exist in the indexed gallery.
        """
        if self.image_embeddings is None or self.text_embeddings is None:
            raise RuntimeError("Index has not been built.")

        img_pos_list: List[int] = []
        txt_pos_list: List[int] = []
        paired_records: List[Dict[str, Any]] = []

        split_norm = split.lower().strip() if split and split.lower() != "all" else None
        if split_norm == "val":
            split_norm = "validation"

        for img_pos, rec_idx in enumerate(self.image_record_indices):
            rec = self.records[rec_idx]
            if split_norm is not None and rec["split"].lower() != split_norm:
                continue
            img_pos_list.append(img_pos)
            txt_pos_list.append(rec_idx)
            paired_records.append(rec)

        if not paired_records:
            return (
                torch.empty((0, self.model.embed_dim), dtype=torch.float32),
                torch.empty((0, self.model.embed_dim), dtype=torch.float32),
                [],
            )

        return (
            self.image_embeddings[img_pos_list],
            self.text_embeddings[txt_pos_list],
            paired_records,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WildMatch AI Member 4 Similarity Matching & Retrieval CLI"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test", "all"],
        help="Gallery split to index (default: test).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="a tiger resting beside shallow water",
        help="Text query for Text-to-Image retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to display (default: 5).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("       WILDMATCH AI — MEMBER 4 RETRIEVAL ENGINE DEMO")
    print("=" * 80)

    retriever = WildMatchRetriever(split=args.split)
    print(
        f"[OK] Indexed split='{args.split}': "
        f"{len(retriever.image_record_indices)} images, "
        f"{len(retriever.records)} captions"
    )

    if len(retriever.image_record_indices) > 0:
        t2i = retriever.retrieve_images_by_text(args.query, top_k=args.top_k)
        print(
            f"\n[Text-to-Image] Query: \"{args.query}\" "
            f"| Predicted Species: {t2i['predicted_species']} "
            f"({t2i['predicted_confidence'] * 100:.1f}%)"
        )
        for item in t2i["results"]:
            print(
                f"  Rank {item['rank']:>2} | Sim: {item['similarity']:+.4f} "
                f"| Species: {item['species']:<14} | {item['image_id']}"
            )

        # Demonstrate Image-to-Text on the top retrieved image
        top_img_path = t2i["results"][0]["image_path"]
        i2t = retriever.retrieve_texts_by_image(top_img_path, top_k=args.top_k)
        print(
            f"\n[Image-to-Text] Query Image: {Path(top_img_path).name} "
            f"| Predicted Species: {i2t['predicted_species']} "
            f"({i2t['predicted_confidence'] * 100:.1f}%)"
        )
        for item in i2t["results"]:
            print(
                f"  Rank {item['rank']:>2} | Sim: {item['similarity']:+.4f} "
                f"| Species: {item['species']:<14} | \"{item['caption']}\""
            )
    else:
        print("[WARN] No image files found under dataset/curated/; skipping image queries.")


if __name__ == "__main__":
    main()

