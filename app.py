"""
WildMatch AI - Member 4: Integrated Gradio Application (`app.py`)

Unifies Member 1's curated wildlife dataset & preprocessing, Member 2/3's
trained Version 3 ResNet-18 + BiLSTM Dual Encoder and Auxiliary Species Heads,
and Member 4's retrieval engine (`retrieval.py`) and evaluation suite (`evaluation.py`)
into an interactive multimodal web application.
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wildmatch_v3 import SPECIES_CLASSES
from retrieval import WildMatchRetriever, resolve_project_path
from evaluation import evaluate_retriever


# ---------------------------------------------------------------------------
# Lazy Singleton Retriever (shared across UI callbacks)
# ---------------------------------------------------------------------------
_RETRIEVER: Optional[WildMatchRetriever] = None


def get_retriever() -> WildMatchRetriever:
    """Initialize and cache the WildMatchRetriever across all 200 curated records."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = WildMatchRetriever(split="all")
    return _RETRIEVER


def _get_sample_images(limit_per_species: int = 1) -> List[List[str]]:
    """Collect representative held-out test image paths for Gradio Examples."""
    retriever = get_retriever()
    seen: Dict[str, int] = {}
    samples: List[List[str]] = []
    # Prefer held-out test split examples
    for rec in retriever.records:
        if rec["split"] != "test" or not rec["has_image_file"]:
            continue
        sp = rec["species"]
        if seen.get(sp, 0) < limit_per_species:
            samples.append([rec["image_path"]])
            seen[sp] = seen.get(sp, 0) + 1
    return samples


# ---------------------------------------------------------------------------
# Tab 1 Callback: Text-to-Image Semantic Retrieval
# ---------------------------------------------------------------------------
def run_text_to_image(
    query_text: str,
    top_k: int,
    split_filter: str,
    species_filter: str,
) -> Tuple[Dict[str, float], str, List[Tuple[str, str]], pd.DataFrame]:
    retriever = get_retriever()
    if not query_text or not query_text.strip():
        empty_df = pd.DataFrame(
            columns=["Rank", "Cosine Sim", "Species", "Pred Species", "Split", "Photo ID", "Caption"]
        )
        return {}, "Please enter a wildlife description to search.", [], empty_df

    res = retriever.retrieve_images_by_text(
        query_text=query_text.strip(),
        top_k=int(top_k),
        species_filter=None if species_filter == "All" else species_filter,
        split_filter=None if split_filter == "all" else split_filter,
    )

    prob_map = res["species_probabilities"]
    unk_note = ""
    if res["unk_token_count"] > 0:
        unk_note = (
            f"\n> **Vocabulary Note:** {res['unk_token_count']} of {res['total_tokens']} "
            "query words mapped to `<UNK>` (158-word training vocabulary). "
            "Try keywords like *tiger, lion, elephant, chital, deer, water, grass, forest, walking, resting*."
        )

    status_md = (
        f"### Query Analysis\n"
        f"- **Predicted Query Species:** `{res['predicted_species']}` "
        f"({res['predicted_confidence'] * 100:.1f}% confidence)\n"
        f"- **Gallery Scope:** `{split_filter}` | **Species Filter:** `{species_filter}` | "
        f"**Returned:** `{len(res['results'])}` matches"
        f"{unk_note}"
    )

    gallery_items: List[Tuple[str, str]] = []
    table_rows: List[Dict[str, Any]] = []
    for item in res["results"]:
        caption_label = (
            f"#{item['rank']} | {item['species']} | Sim: {item['similarity']:+.4f} "
            f"[{item['split']}]\n{item['caption']}"
        )
        gallery_items.append((item["image_path"], caption_label))
        table_rows.append(
            {
                "Rank": item["rank"],
                "Cosine Sim": round(item["similarity"], 4),
                "Species": item["species"],
                "Pred Species": f"{item['predicted_species']} ({item['predicted_confidence']*100:.1f}%)",
                "Split": item["split"],
                "Photo ID": item["photo_id"],
                "Caption": item["caption"],
            }
        )

    df_out = pd.DataFrame(table_rows)
    return prob_map, status_md, gallery_items, df_out


# ---------------------------------------------------------------------------
# Tab 2 Callback: Image-to-Text Semantic Retrieval
# ---------------------------------------------------------------------------
def run_image_to_text(
    image: Optional[Image.Image],
    top_k: int,
    split_filter: str,
    species_filter: str,
) -> Tuple[Dict[str, float], str, pd.DataFrame]:
    retriever = get_retriever()
    if image is None:
        empty_df = pd.DataFrame(
            columns=["Rank", "Cosine Sim", "Species", "Pred Text Species", "Split", "Photo ID", "Caption"]
        )
        return {}, "Please upload or select a wildlife image.", empty_df

    res = retriever.retrieve_texts_by_image(
        image_input=image,
        top_k=int(top_k),
        species_filter=None if species_filter == "All" else species_filter,
        split_filter=None if split_filter == "all" else split_filter,
    )

    prob_map = res["species_probabilities"]
    top_caption = res["results"][0]["caption"] if res["results"] else "N/A"
    top_sim = f"{res['results'][0]['similarity']:+.4f}" if res["results"] else "N/A"

    status_md = (
        f"### Image Analysis\n"
        f"- **Predicted Image Species:** `{res['predicted_species']}` "
        f"({res['predicted_confidence'] * 100:.1f}% confidence)\n"
        f"- **Best Matching Caption (Sim {top_sim}):** *\"{top_caption}\"*\n"
        f"- **Gallery Scope:** `{split_filter}` | **Species Filter:** `{species_filter}`"
    )

    table_rows: List[Dict[str, Any]] = []
    for item in res["results"]:
        table_rows.append(
            {
                "Rank": item["rank"],
                "Cosine Sim": round(item["similarity"], 4),
                "Species": item["species"],
                "Pred Text Species": f"{item['predicted_species']} ({item['predicted_confidence']*100:.1f}%)",
                "Split": item["split"],
                "Photo ID": item["photo_id"],
                "Caption": item["caption"],
            }
        )

    return prob_map, status_md, pd.DataFrame(table_rows)


# ---------------------------------------------------------------------------
# Tab 3 Callback: Cross-Modal Pair Similarity Scorer
# ---------------------------------------------------------------------------
def run_pair_scorer(
    image: Optional[Image.Image],
    caption: str,
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    retriever = get_retriever()
    if image is None:
        return "Please upload or select an image.", {}, {}
    if not caption or not caption.strip():
        return "Please enter a caption to compare.", {}, {}

    res = retriever.score_image_text_pair(image_input=image, caption=caption.strip())
    agreement_str = "Match (Same Species)" if res["species_agreement"] else "Mismatch (Different Species)"
    sim = res["cosine_similarity"]

    if sim >= 0.45:
        strength = "Strong Semantic Match"
    elif sim >= 0.15:
        strength = "Moderate Semantic Alignment"
    else:
        strength = "Low / Dissimilar Pair"

    summary_md = (
        f"### Cross-Modal Similarity Score\n"
        f"- **Cosine Similarity:** `{sim:+.4f}` (**{strength}**)\n"
        f"- **Temperature-Scaled Logit:** `{res['scaled_logit']:+.2f}`\n"
        f"- **Image Predicted Species:** `{res['image_predicted_species']}` "
        f"({res['image_confidence'] * 100:.1f}%)\n"
        f"- **Text Predicted Species:** `{res['text_predicted_species']}` "
        f"({res['text_confidence'] * 100:.1f}%)\n"
        f"- **Auxiliary Species Agreement:** `{agreement_str}`"
    )
    return summary_md, res["image_species_probabilities"], res["text_species_probabilities"]


# ---------------------------------------------------------------------------
# Tab 4 Callback: Evaluation & Benchmarks
# ---------------------------------------------------------------------------
def run_evaluation_ui(split_name: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    retriever = get_retriever()
    ev = evaluate_retriever(retriever, split=split_name)
    m = ev["metrics"]
    s = ev["similarity_stats"]

    headline_md = (
        f"### Evaluation Summary — `{split_name.upper()}` Split ($N = {ev['num_samples']}$)\n"
        f"- **Species Retrieval (Primary):** "
        f"T2I R@1 = **{m['t2i_species_r1']*100:.2f}%**, MRR = **{m['t2i_species_mrr']:.4f}** | "
        f"I2T R@1 = **{m['i2t_species_r1']*100:.2f}%**, R@5 = **{m['i2t_species_r5']*100:.2f}%**, "
        f"MRR = **{m['i2t_species_mrr']:.4f}**\n"
        f"- **Exact-Pair Retrieval:** "
        f"T2I R@5 = **{m['t2i_exact_r5']*100:.2f}%**, MRR = **{m['t2i_exact_mrr']:.4f}** | "
        f"I2T R@5 = **{m['i2t_exact_r5']*100:.2f}%**, MRR = **{m['i2t_exact_mrr']:.4f}**\n"
        f"- **Auxiliary Classifiers:** "
        f"Image Acc = **{m['image_classifier_accuracy']*100:.2f}%**, "
        f"Text Acc = **{m['text_classifier_accuracy']*100:.2f}%**\n"
        f"- **Cosine Geometry:** "
        f"Same-Species Mean = `{s['mean_same_species_similarity']:+.4f}`, "
        f"Diff-Species Mean = `{s['mean_diff_species_similarity']:+.4f}`, "
        f"Separation Margin = **`{s['species_separation_margin']:+.4f}`**"
    )

    overall_rows = [
        {
            "Task": "Text-to-Image (Species)",
            "Recall@1": f"{m['t2i_species_r1']*100:.2f}%",
            "Recall@5": f"{m['t2i_species_r5']*100:.2f}%",
            "Recall@10": f"{m['t2i_species_r10']*100:.2f}%",
            "MRR": round(m["t2i_species_mrr"], 4),
        },
        {
            "Task": "Image-to-Text (Species)",
            "Recall@1": f"{m['i2t_species_r1']*100:.2f}%",
            "Recall@5": f"{m['i2t_species_r5']*100:.2f}%",
            "Recall@10": f"{m['i2t_species_r10']*100:.2f}%",
            "MRR": round(m["i2t_species_mrr"], 4),
        },
        {
            "Task": "Bimodal Average (Species)",
            "Recall@1": f"{m['species_r1_average']*100:.2f}%",
            "Recall@5": f"{m['species_r5_average']*100:.2f}%",
            "Recall@10": f"{m['species_r10_average']*100:.2f}%",
            "MRR": round(m["species_mrr_average"], 4),
        },
        {
            "Task": "Text-to-Image (Exact Pair)",
            "Recall@1": f"{m['t2i_exact_r1']*100:.2f}%",
            "Recall@5": f"{m['t2i_exact_r5']*100:.2f}%",
            "Recall@10": f"{m['t2i_exact_r10']*100:.2f}%",
            "MRR": round(m["t2i_exact_mrr"], 4),
        },
        {
            "Task": "Image-to-Text (Exact Pair)",
            "Recall@1": f"{m['i2t_exact_r1']*100:.2f}%",
            "Recall@5": f"{m['i2t_exact_r5']*100:.2f}%",
            "Recall@10": f"{m['i2t_exact_r10']*100:.2f}%",
            "MRR": round(m["i2t_exact_mrr"], 4),
        },
    ]

    species_rows = []
    for sp, sp_m in ev["per_species"].items():
        species_rows.append(
            {
                "Species": sp,
                "Samples": sp_m["count"],
                "T2I Sp R@1": f"{sp_m['t2i_species_r1']*100:.1f}%",
                "T2I Sp MRR": round(sp_m["t2i_species_mrr"], 4),
                "I2T Sp R@1": f"{sp_m['i2t_species_r1']*100:.1f}%",
                "I2T Sp R@5": f"{sp_m['i2t_species_r5']*100:.1f}%",
                "I2T Sp MRR": round(sp_m["i2t_species_mrr"], 4),
                "T2I Exact R@5": f"{sp_m['t2i_exact_r5']*100:.1f}%",
                "I2T Exact R@5": f"{sp_m['i2t_exact_r5']*100:.1f}%",
            }
        )

    return headline_md, pd.DataFrame(overall_rows), pd.DataFrame(species_rows)


# ---------------------------------------------------------------------------
# Gradio Interface Builder
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    retriever = get_retriever()
    sample_images = _get_sample_images(limit_per_species=1)

    loss_plot_path = resolve_project_path("models/wildmatch_v3_loss.png")
    val_plot_path = resolve_project_path("models/wildmatch_v3_validation.png")
    img_cm_path = resolve_project_path("models/wildmatch_v3_image_confusion.png")
    txt_cm_path = resolve_project_path("models/wildmatch_v3_text_confusion.png")

    with gr.Blocks(title="WildMatch AI — Wildlife Semantic Retrieval") as demo:
        gr.Markdown(
            f"""
            # WildMatch AI — Multimodal Wildlife Semantic Matching & Retrieval
            **Architecture:** Fine-Tuned ResNet-18 Image Encoder + BiLSTM Text Encoder ($D=256$ Shared Unit Hypersphere)  
            **Focal Species:** Asian Elephant, Chital, Lion, Tiger | **Indexed Gallery:** {len(retriever.image_record_indices)} Wildlife Images & {len(retriever.records)} Captions
            """
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Text-to-Image Retrieval
            # ---------------------------------------------------------------
            with gr.Tab("Text-to-Image Retrieval"):
                gr.Markdown(
                    "Search the curated wildlife gallery using natural language descriptions. "
                    "Supports Top-5 / Top-10 ranking and species filtering."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        t2i_query = gr.Textbox(
                            label="Wildlife Description Query",
                            placeholder="e.g., a tiger resting near shallow water in the jungle",
                            value="a tiger resting beside shallow water",
                            lines=2,
                        )
                        with gr.Row():
                            t2i_topk = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1, label="Top-K Results"
                            )
                            t2i_split = gr.Dropdown(
                                choices=["all", "test", "validation", "train"],
                                value="all",
                                label="Gallery Split",
                            )
                            t2i_species = gr.Dropdown(
                                choices=["All"] + list(SPECIES_CLASSES),
                                value="All",
                                label="Species Filter",
                            )
                        t2i_btn = gr.Button("Search Wildlife Images", variant="primary")
                        gr.Examples(
                            examples=[
                                ["a tiger walking through tall grass in the wild"],
                                ["two lions lying in dry grass in open field"],
                                ["an asian elephant swimming in a lake with its trunk raised"],
                                ["two chital deer grazing in the woods"],
                            ],
                            inputs=[t2i_query],
                            label="Suggested In-Vocabulary Queries",
                        )
                    with gr.Column(scale=1):
                        t2i_probs = gr.Label(
                            num_top_classes=4,
                            label="Query Text Species Prediction",
                        )
                        t2i_status = gr.Markdown()

                t2i_gallery = gr.Gallery(
                    label="Top-K Matching Wildlife Images",
                    columns=5,
                    height="auto",
                    object_fit="cover",
                )
                t2i_table = gr.Dataframe(
                    label="Ranked Similarity Scores & Metadata",
                    interactive=False,
                )

                t2i_btn.click(
                    fn=run_text_to_image,
                    inputs=[t2i_query, t2i_topk, t2i_split, t2i_species],
                    outputs=[t2i_probs, t2i_status, t2i_gallery, t2i_table],
                )

            # ---------------------------------------------------------------
            # Tab 2: Image-to-Text Retrieval
            # ---------------------------------------------------------------
            with gr.Tab("Image-to-Text Retrieval"):
                gr.Markdown(
                    "Upload a wildlife photo (or select a held-out test sample below) to predict its species "
                    "and retrieve the most semantically aligned descriptions."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        i2t_image = gr.Image(
                            type="pil",
                            label="Query Wildlife Image",
                            height=320,
                        )
                        if sample_images:
                            gr.Examples(
                                examples=sample_images,
                                inputs=[i2t_image],
                                label="Held-Out Test Set Samples (One per Species)",
                            )
                    with gr.Column(scale=1):
                        with gr.Row():
                            i2t_topk = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1, label="Top-K Captions"
                            )
                            i2t_split = gr.Dropdown(
                                choices=["all", "test", "validation", "train"],
                                value="all",
                                label="Caption Gallery Split",
                            )
                            i2t_species = gr.Dropdown(
                                choices=["All"] + list(SPECIES_CLASSES),
                                value="All",
                                label="Species Filter",
                            )
                        i2t_btn = gr.Button("Retrieve Matching Captions", variant="primary")
                        i2t_probs = gr.Label(
                            num_top_classes=4,
                            label="Predicted Image Species",
                        )
                        i2t_status = gr.Markdown()

                i2t_table = gr.Dataframe(
                    label="Top-K Matching Captions",
                    interactive=False,
                )

                i2t_btn.click(
                    fn=run_image_to_text,
                    inputs=[i2t_image, i2t_topk, i2t_split, i2t_species],
                    outputs=[i2t_probs, i2t_status, i2t_table],
                )

            # ---------------------------------------------------------------
            # Tab 3: Cross-Modal Pair Matcher
            # ---------------------------------------------------------------
            with gr.Tab("Pair Similarity Scorer"):
                gr.Markdown(
                    "Directly measure cosine similarity in $\\mathbb{S}^{255}$ between any wildlife image "
                    "and any custom caption."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        pair_image = gr.Image(type="pil", label="Wildlife Image", height=300)
                        if sample_images:
                            gr.Examples(
                                examples=sample_images,
                                inputs=[pair_image],
                                label="Sample Test Images",
                            )
                    with gr.Column(scale=1):
                        pair_caption = gr.Textbox(
                            label="Caption to Compare",
                            value="a tiger walking through dense brush",
                            lines=2,
                        )
                        pair_btn = gr.Button("Compute Cosine Similarity", variant="primary")
                        pair_summary = gr.Markdown()
                        with gr.Row():
                            pair_img_probs = gr.Label(num_top_classes=4, label="Image Species")
                            pair_txt_probs = gr.Label(num_top_classes=4, label="Text Species")

                pair_btn.click(
                    fn=run_pair_scorer,
                    inputs=[pair_image, pair_caption],
                    outputs=[pair_summary, pair_img_probs, pair_txt_probs],
                )

            # ---------------------------------------------------------------
            # Tab 4: Evaluation Metrics & Training Analytics
            # ---------------------------------------------------------------
            with gr.Tab("Evaluation & Analytics"):
                gr.Markdown(
                    "Live retrieval benchmark computing **Recall@1**, **Recall@5**, **Recall@10**, "
                    "**Mean Reciprocal Rank (MRR)**, and **Cosine Separation Margin**, alongside Member 3 training plots."
                )
                with gr.Row():
                    eval_split = gr.Radio(
                        choices=["test", "validation", "train", "all"],
                        value="test",
                        label="Evaluation Split (Test = 30 Held-Out Samples)",
                    )
                    eval_btn = gr.Button("Run Evaluation Suite", variant="primary")

                eval_headline = gr.Markdown()
                eval_overall_df = gr.Dataframe(label="Overall Retrieval Metrics", interactive=False)
                eval_species_df = gr.Dataframe(label="Per-Species Breakdown", interactive=False)

                gr.Markdown("### Member 3 Training Curves & Test Confusion Matrices")
                with gr.Row():
                    if loss_plot_path.is_file():
                        gr.Image(value=str(loss_plot_path), label="Training Loss Curves", interactive=False)
                    if val_plot_path.is_file():
                        gr.Image(value=str(val_plot_path), label="Validation Retrieval Curves", interactive=False)
                with gr.Row():
                    if img_cm_path.is_file():
                        gr.Image(value=str(img_cm_path), label="Test Image Confusion Matrix", interactive=False)
                    if txt_cm_path.is_file():
                        gr.Image(value=str(txt_cm_path), label="Test Text Confusion Matrix", interactive=False)

                eval_btn.click(
                    fn=run_evaluation_ui,
                    inputs=[eval_split],
                    outputs=[eval_headline, eval_overall_df, eval_species_df],
                )

                # Populate test split metrics automatically on initial page load
                demo.load(
                    fn=run_evaluation_ui,
                    inputs=[eval_split],
                    outputs=[eval_headline, eval_overall_df, eval_species_df],
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch WildMatch AI Gradio Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        allowed_paths=[str(PROJECT_ROOT)],
    )


if __name__ == "__main__":
    main()
