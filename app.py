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
import html
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
# Design System: Theme & Accessible Custom CSS
# ---------------------------------------------------------------------------
APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_md,
    font=[
        "Segoe UI",
        "system-ui",
        "-apple-system",
        "sans-serif",
    ],
    font_mono=[
        "Consolas",
        "ui-monospace",
        "SFMono-Regular",
        "monospace",
    ],
)

CUSTOM_CSS = """
:root {
    --wm-surface: #ffffff;
    --wm-surface-subtle: #f8fafc;
    --wm-border: #cbd5e1;
    --wm-text: #0f172a;
    --wm-muted: #475569;
    --wm-primary: #15803d;
    --wm-primary-soft: #dcfce7;
    --wm-primary-text: #14532d;
    --wm-navy: #1e3a5f;
    --wm-accent: #b45309;
    --wm-accent-soft: #fef3c7;
    --wm-danger: #b91c1c;
    --wm-danger-soft: #fee2e2;
    --wm-focus: #15803d;
}

.dark, [data-theme="dark"] {
    --wm-surface: #0f172a;
    --wm-surface-subtle: #1e293b;
    --wm-border: #334155;
    --wm-text: #f8fafc;
    --wm-muted: #cbd5e1;
    --wm-primary: #22c55e;
    --wm-primary-soft: rgba(34, 197, 94, 0.16);
    --wm-primary-text: #bbf7d0;
    --wm-navy: #93c5fd;
    --wm-accent: #f59e0b;
    --wm-accent-soft: rgba(245, 158, 11, 0.18);
    --wm-danger: #f87171;
    --wm-danger-soft: rgba(248, 113, 113, 0.18);
    --wm-focus: #4ade80;
}

.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
}

/* Interactive cursor & WCAG-compliant focus rings */
button,
[role="tab"],
select,
.gallery-item,
.examples-holder button,
.examples-holder tr {
    cursor: pointer;
}

button:focus-visible,
[role="tab"]:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
a:focus-visible {
    outline: 2px solid var(--wm-focus) !important;
    outline-offset: 2px !important;
}

/* Hero Banner */
.wm-hero {
    background: linear-gradient(135deg, var(--wm-surface) 0%, var(--wm-surface-subtle) 100%);
    border: 1px solid var(--wm-border);
    border-left: 5px solid var(--wm-primary);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    color: var(--wm-text);
}

.wm-hero-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}

.wm-hero-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.wm-hero-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: var(--wm-primary-soft);
    color: var(--wm-primary-text);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.wm-hero-title {
    margin: 0;
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.2;
    color: var(--wm-text);
}

.wm-hero-subtitle {
    margin: 4px 0 0 0;
    font-size: 0.92rem;
    color: var(--wm-muted);
}

.wm-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.wm-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 11px;
    border-radius: 999px;
    border: 1px solid var(--wm-border);
    background: var(--wm-surface);
    color: var(--wm-text);
    font-size: 0.8rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

.wm-pill strong {
    color: var(--wm-primary-text);
    font-family: var(--font-mono);
}

/* KPI Grid for Evaluation Tab */
.wm-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin: 8px 0 12px 0;
}

.wm-kpi-card {
    background: var(--wm-surface);
    border: 1px solid var(--wm-border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 150ms ease, box-shadow 150ms ease;
}

.wm-kpi-card:hover {
    border-color: var(--wm-primary);
}

.wm-kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--wm-muted);
    margin-bottom: 6px;
}

.wm-kpi-value {
    font-family: var(--font-mono);
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.15;
    color: var(--wm-text);
    font-variant-numeric: tabular-nums;
}

.wm-kpi-sub {
    margin-top: 5px;
    font-size: 0.8rem;
    color: var(--wm-muted);
    font-variant-numeric: tabular-nums;
}

/* Similarity Meter Card for Pair Scorer */
.wm-meter-card {
    background: var(--wm-surface);
    border: 1px solid var(--wm-border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}

.wm-meter-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
}

.wm-meter-score {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--wm-text);
    font-variant-numeric: tabular-nums;
}

.wm-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.wm-badge-match {
    background: var(--wm-primary-soft);
    color: var(--wm-primary-text);
}

.wm-badge-warn {
    background: var(--wm-accent-soft);
    color: var(--wm-accent);
}

.wm-badge-mismatch {
    background: var(--wm-danger-soft);
    color: var(--wm-danger);
}

.wm-meter-track {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: var(--wm-surface-subtle);
    border: 1px solid var(--wm-border);
    overflow: hidden;
}

.wm-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 220ms ease;
}

.wm-meter-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--wm-muted);
}

.wm-alert-empty {
    background: var(--wm-surface-subtle);
    border: 1px dashed var(--wm-border);
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--wm-muted);
    font-size: 0.9rem;
}

.wm-section-note {
    font-size: 0.86rem;
    color: var(--wm-muted);
    margin-bottom: 6px;
}

@media (max-width: 768px) {
    .wm-hero {
        padding: 14px 16px;
    }
    .wm-hero-title {
        font-size: 1.2rem;
    }
    .wm-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 480px) {
    .wm-kpi-grid {
        grid-template-columns: 1fr;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
    }
}
"""


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


def _get_sample_images(limit_per_species: int = 1) -> Tuple[List[List[str]], List[str]]:
    """Collect representative held-out test image paths and readable labels for Gradio Examples."""
    retriever = get_retriever()
    seen: Dict[str, int] = {}
    samples: List[List[str]] = []
    labels: List[str] = []
    for rec in retriever.records:
        if rec["split"] != "test" or not rec["has_image_file"]:
            continue
        sp = rec["species"]
        if seen.get(sp, 0) < limit_per_species:
            samples.append([rec["image_path"]])
            labels.append(f"{sp} (Test)")
            seen[sp] = seen.get(sp, 0) + 1
    return samples, labels


def _empty_t2i_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Rank", "Cosine Sim", "Species", "Pred Species", "Split", "Photo ID", "Caption"]
    )


def _empty_i2t_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Rank", "Cosine Sim", "Species", "Pred Text Species", "Split", "Photo ID", "Caption"]
    )


# ---------------------------------------------------------------------------
# HTML Visual Builders (All dynamic values escaped)
# ---------------------------------------------------------------------------
def _build_hero_html(num_images: int, num_captions: int, vocab_size: int) -> str:
    return f"""
    <header class="wm-hero" role="banner">
      <div class="wm-hero-header">
        <div class="wm-hero-brand">
          <div class="wm-hero-icon" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <path d="m21 21-4.3-4.3"></path>
              <path d="M11 8a3 3 0 0 0-3 3"></path>
            </svg>
          </div>
          <div>
            <h1 class="wm-hero-title">WildMatch AI — Wildlife Semantic Retrieval</h1>
            <p class="wm-hero-subtitle">
              Fine-Tuned ResNet-18 Image Encoder + BiLSTM Text Encoder in a 256-D Unit Hypersphere
            </p>
          </div>
        </div>
        <div class="wm-pill-row" aria-label="System index summary">
          <span class="wm-pill">Gallery <strong>{num_images}</strong> imgs / <strong>{num_captions}</strong> captions</span>
          <span class="wm-pill">Species <strong>4</strong> (Elephant, Chital, Lion, Tiger)</span>
          <span class="wm-pill">Vocab <strong>{vocab_size}</strong> tokens</span>
        </div>
      </div>
    </header>
    """


def _build_pair_meter_html(
    sim: Optional[float] = None,
    scaled_logit: Optional[float] = None,
    img_sp: str = "",
    img_conf: float = 0.0,
    txt_sp: str = "",
    txt_conf: float = 0.0,
    agreement: Optional[bool] = None,
    message: Optional[str] = None,
) -> str:
    if sim is None:
        safe_msg = html.escape(message or "Upload or choose a sample image and enter a caption to compute cosine similarity.")
        return f'<div class="wm-alert-empty" role="status" aria-live="polite">{safe_msg}</div>'

    pct = max(0.0, min(100.0, (sim + 1.0) * 50.0))
    if sim >= 0.45:
        strength_label = "Strong Semantic Match"
        badge_cls = "wm-badge-match"
        fill_color = "#15803d"
    elif sim >= 0.15:
        strength_label = "Moderate Alignment"
        badge_cls = "wm-badge-warn"
        fill_color = "#d97706"
    else:
        strength_label = "Low / Dissimilar Pair"
        badge_cls = "wm-badge-mismatch"
        fill_color = "#dc2626"

    agree_badge_cls = "wm-badge-match" if agreement else "wm-badge-mismatch"
    agree_label = "Species Match" if agreement else "Species Mismatch"
    safe_img_sp = html.escape(img_sp)
    safe_txt_sp = html.escape(txt_sp)
    logit_str = f"{scaled_logit:+.2f}" if scaled_logit is not None else "N/A"

    return f"""
    <div class="wm-meter-card" role="region" aria-label="Cross-modal similarity score" aria-live="polite">
      <div class="wm-meter-top">
        <div>
          <div class="wm-kpi-label">Cosine Similarity</div>
          <div class="wm-meter-score">{sim:+.4f}</div>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <span class="wm-badge {badge_cls}">{strength_label}</span>
          <span class="wm-badge {agree_badge_cls}">{agree_label}</span>
        </div>
      </div>
      <div class="wm-meter-track" role="progressbar" aria-valuenow="{sim:.2f}" aria-valuemin="-1" aria-valuemax="1">
        <div class="wm-meter-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>
      </div>
      <div class="wm-meter-scale">
        <span>-1.0 (Opposite)</span>
        <span>0.0 (Orthogonal)</span>
        <span>+1.0 (Aligned)</span>
      </div>
      <div class="wm-kpi-sub" style="margin-top:10px;">
        <strong>Logit:</strong> {logit_str} &nbsp;|&nbsp;
        <strong>Image:</strong> {safe_img_sp} ({img_conf * 100:.1f}%) &nbsp;|&nbsp;
        <strong>Text:</strong> {safe_txt_sp} ({txt_conf * 100:.1f}%)
      </div>
    </div>
    """


def _build_eval_kpis_html(ev: Dict[str, Any]) -> str:
    m = ev["metrics"]
    s = ev["similarity_stats"]
    split_label = html.escape(str(ev.get("split", "test")).upper())
    n_samples = int(ev.get("num_samples", 0))
    return f"""
    <div class="wm-kpi-grid" role="region" aria-label="Evaluation key performance indicators">
      <div class="wm-kpi-card">
        <div class="wm-kpi-label">T2I Species R@1 ({split_label})</div>
        <div class="wm-kpi-value">{m['t2i_species_r1'] * 100:.1f}%</div>
        <div class="wm-kpi-sub">MRR {m['t2i_species_mrr']:.4f} · R@5 {m['t2i_species_r5'] * 100:.1f}%</div>
      </div>
      <div class="wm-kpi-card">
        <div class="wm-kpi-label">I2T Species R@1 ({split_label})</div>
        <div class="wm-kpi-value">{m['i2t_species_r1'] * 100:.1f}%</div>
        <div class="wm-kpi-sub">MRR {m['i2t_species_mrr']:.4f} · R@5 {m['i2t_species_r5'] * 100:.1f}%</div>
      </div>
      <div class="wm-kpi-card">
        <div class="wm-kpi-label">Cosine Separation Margin</div>
        <div class="wm-kpi-value">{s['species_separation_margin']:+.4f}</div>
        <div class="wm-kpi-sub">Same {s['mean_same_species_similarity']:+.3f} vs Diff {s['mean_diff_species_similarity']:+.3f}</div>
      </div>
      <div class="wm-kpi-card">
        <div class="wm-kpi-label">Classifier Acc (N={n_samples})</div>
        <div class="wm-kpi-value">{m['image_classifier_accuracy'] * 100:.1f}% / {m['text_classifier_accuracy'] * 100:.1f}%</div>
        <div class="wm-kpi-sub">Image Head / Text Head Accuracy</div>
      </div>
    </div>
    """


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
        return (
            {},
            "### Enter a Wildlife Description\n"
            "Type a phrase (e.g., *a tiger walking through tall grass*) or click a suggested query below to search.",
            [],
            _empty_t2i_df(),
        )

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

    if not res["results"]:
        status_md = (
            f"### No Matching Images Found\n"
            f"- **Predicted Query Species:** `{res['predicted_species']}` "
            f"({res['predicted_confidence'] * 100:.1f}% confidence)\n"
            f"- **Active Filters:** Split `{split_filter}` | Species `{species_filter}`\n"
            f"- **Suggestion:** Set **Gallery Split** to `all` or **Species Filter** to `All` to broaden results."
            f"{unk_note}"
        )
        return prob_map, status_md, [], _empty_t2i_df()

    top_sim = res["results"][0]["similarity"]
    status_md = (
        f"### Query Analysis\n"
        f"- **Predicted Query Species:** `{res['predicted_species']}` "
        f"({res['predicted_confidence'] * 100:.1f}% confidence)\n"
        f"- **Top Match Score:** `{top_sim:+.4f}` | **Returned:** `{len(res['results'])}` matches\n"
        f"- **Gallery Scope:** `{split_filter}` | **Species Filter:** `{species_filter}`"
        f"{unk_note}"
    )

    gallery_items: List[Tuple[str, str]] = []
    table_rows: List[Dict[str, Any]] = []
    for item in res["results"]:
        caption_label = (
            f"#{item['rank']} · {item['species']} · Sim {item['similarity']:+.4f} "
            f"[{item['split']}]\n{item['caption']}"
        )
        gallery_items.append((item["image_path"], caption_label))
        table_rows.append(
            {
                "Rank": item["rank"],
                "Cosine Sim": round(item["similarity"], 4),
                "Species": item["species"],
                "Pred Species": f"{item['predicted_species']} ({item['predicted_confidence'] * 100:.1f}%)",
                "Split": item["split"],
                "Photo ID": item["photo_id"],
                "Caption": item["caption"],
            }
        )

    return prob_map, status_md, gallery_items, pd.DataFrame(table_rows)


def _refresh_t2i_on_filter(
    query_text: str,
    top_k: int,
    split_filter: str,
    species_filter: str,
) -> Tuple[Dict[str, float], str, List[Tuple[str, str]], pd.DataFrame]:
    if not query_text or not query_text.strip():
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return run_text_to_image(query_text, top_k, split_filter, species_filter)


def _clear_t2i() -> Tuple[str, int, str, str, Dict[str, float], str, List[Tuple[str, str]], pd.DataFrame]:
    return (
        "",
        5,
        "all",
        "All",
        {},
        "### Ready to Search\nEnter a wildlife description or click any suggested query chip to search the gallery.",
        [],
        _empty_t2i_df(),
    )


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
        return (
            {},
            "### Upload or Select a Wildlife Photo\n"
            "Drop an image on the left or click one of the held-out test samples below to retrieve matching captions.",
            _empty_i2t_df(),
        )

    res = retriever.retrieve_texts_by_image(
        image_input=image,
        top_k=int(top_k),
        species_filter=None if species_filter == "All" else species_filter,
        split_filter=None if split_filter == "all" else split_filter,
    )

    prob_map = res["species_probabilities"]
    if not res["results"]:
        status_md = (
            f"### No Matching Captions Found\n"
            f"- **Predicted Image Species:** `{res['predicted_species']}` "
            f"({res['predicted_confidence'] * 100:.1f}% confidence)\n"
            f"- **Suggestion:** Set **Caption Gallery Split** to `all` or **Species Filter** to `All`."
        )
        return prob_map, status_md, _empty_i2t_df()

    top_caption = res["results"][0]["caption"]
    top_sim = f"{res['results'][0]['similarity']:+.4f}"

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
                "Pred Text Species": f"{item['predicted_species']} ({item['predicted_confidence'] * 100:.1f}%)",
                "Split": item["split"],
                "Photo ID": item["photo_id"],
                "Caption": item["caption"],
            }
        )

    return prob_map, status_md, pd.DataFrame(table_rows)


def _refresh_i2t_on_filter(
    image: Optional[Image.Image],
    top_k: int,
    split_filter: str,
    species_filter: str,
) -> Tuple[Dict[str, float], str, pd.DataFrame]:
    if image is None:
        return gr.skip(), gr.skip(), gr.skip()
    return run_image_to_text(image, top_k, split_filter, species_filter)


def _clear_i2t() -> Tuple[None, int, str, str, Dict[str, float], str, pd.DataFrame]:
    return (
        None,
        5,
        "all",
        "All",
        {},
        "### Ready for Image Query\nUpload a photo or select a held-out test sample to retrieve top matching captions.",
        _empty_i2t_df(),
    )


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


def _run_pair_scorer_ui(
    image: Optional[Image.Image],
    caption: str,
) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
    if image is None:
        msg = "Please upload or select a wildlife image to compare against the caption."
        return _build_pair_meter_html(message=msg), msg, {}, {}
    if not caption or not caption.strip():
        msg = "Please enter a wildlife caption to compare against the selected image."
        return _build_pair_meter_html(message=msg), msg, {}, {}

    retriever = get_retriever()
    res = retriever.score_image_text_pair(image_input=image, caption=caption.strip())
    summary_md, img_probs, txt_probs = run_pair_scorer(image, caption)
    meter_html = _build_pair_meter_html(
        sim=res["cosine_similarity"],
        scaled_logit=res["scaled_logit"],
        img_sp=res["image_predicted_species"],
        img_conf=res["image_confidence"],
        txt_sp=res["text_predicted_species"],
        txt_conf=res["text_confidence"],
        agreement=res["species_agreement"],
    )
    return meter_html, summary_md, img_probs, txt_probs


def _clear_pair() -> Tuple[None, str, str, str, Dict[str, float], Dict[str, float]]:
    msg = "Upload or choose a sample image and enter a caption to compute cosine similarity."
    return None, "", _build_pair_meter_html(message=msg), msg, {}, {}


# ---------------------------------------------------------------------------
# Tab 4 Callback: Evaluation & Benchmarks
# ---------------------------------------------------------------------------
def _format_eval_outputs(ev: Dict[str, Any], split_name: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
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


def run_evaluation_ui(split_name: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    retriever = get_retriever()
    ev = evaluate_retriever(retriever, split=split_name)
    return _format_eval_outputs(ev, split_name)


def _run_evaluation_dashboard(split_name: str) -> Tuple[str, str, pd.DataFrame, pd.DataFrame]:
    retriever = get_retriever()
    ev = evaluate_retriever(retriever, split=split_name)
    headline_md, overall_df, species_df = _format_eval_outputs(ev, split_name)
    kpis_html = _build_eval_kpis_html(ev)
    return kpis_html, headline_md, overall_df, species_df


# ---------------------------------------------------------------------------
# Gradio Interface Builder
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    retriever = get_retriever()
    sample_images, sample_labels = _get_sample_images(limit_per_species=1)
    default_sample_path = sample_images[0][0] if sample_images else None
    vocab_size = len(retriever.vocab) if hasattr(retriever, "vocab") else 158

    loss_plot_path = resolve_project_path("models/wildmatch_v3_loss.png")
    val_plot_path = resolve_project_path("models/wildmatch_v3_validation.png")
    img_cm_path = resolve_project_path("models/wildmatch_v3_image_confusion.png")
    txt_cm_path = resolve_project_path("models/wildmatch_v3_text_confusion.png")

    with gr.Blocks(title="WildMatch AI — Wildlife Semantic Retrieval") as demo:
        gr.HTML(
            _build_hero_html(
                num_images=len(retriever.image_record_indices),
                num_captions=len(retriever.records),
                vocab_size=vocab_size,
            )
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Text-to-Image Retrieval
            # ---------------------------------------------------------------
            with gr.Tab("Text-to-Image Retrieval"):
                gr.Markdown(
                    "Search the curated wildlife gallery using natural language descriptions. "
                    "Press **Enter** or click **Search Wildlife Images**, or click any suggested query to run immediately."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
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
                        with gr.Row():
                            t2i_btn = gr.Button("Search Wildlife Images", variant="primary", scale=3)
                            t2i_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                    with gr.Column(scale=5):
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
                    preview=True,
                )

                with gr.Accordion("Ranked Similarity Scores & Metadata", open=True):
                    t2i_table = gr.Dataframe(
                        label="Ranked Results Table",
                        interactive=False,
                        wrap=True,
                        max_height=360,
                        show_search="search",
                    )

                gr.Examples(
                    examples=[
                        ["a tiger walking through tall grass in the wild"],
                        ["two lions lying in dry grass in open field"],
                        ["an asian elephant swimming in a lake with its trunk raised"],
                        ["two chital deer grazing in the woods"],
                    ],
                    inputs=[t2i_query],
                    label="Suggested In-Vocabulary Queries (Click to Load)",
                )

                t2i_inputs = [t2i_query, t2i_topk, t2i_split, t2i_species]
                t2i_outputs = [t2i_probs, t2i_status, t2i_gallery, t2i_table]

                t2i_btn.click(fn=run_text_to_image, inputs=t2i_inputs, outputs=t2i_outputs)
                t2i_query.submit(fn=run_text_to_image, inputs=t2i_inputs, outputs=t2i_outputs)
                t2i_topk.release(fn=_refresh_t2i_on_filter, inputs=t2i_inputs, outputs=t2i_outputs)
                t2i_split.change(fn=_refresh_t2i_on_filter, inputs=t2i_inputs, outputs=t2i_outputs)
                t2i_species.change(fn=_refresh_t2i_on_filter, inputs=t2i_inputs, outputs=t2i_outputs)
                t2i_clear_btn.click(
                    fn=_clear_t2i,
                    inputs=None,
                    outputs=[t2i_query, t2i_topk, t2i_split, t2i_species] + t2i_outputs,
                )

            # ---------------------------------------------------------------
            # Tab 2: Image-to-Text Retrieval
            # ---------------------------------------------------------------
            with gr.Tab("Image-to-Text Retrieval"):
                gr.Markdown(
                    "Upload a wildlife photo or click a held-out test sample below to predict its species "
                    "and retrieve the most semantically aligned descriptions."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        i2t_image = gr.Image(
                            type="pil",
                            value=default_sample_path,
                            label="Query Wildlife Image",
                            height=320,
                        )
                    with gr.Column(scale=7):
                        i2t_topk = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1, label="Top-K Captions"
                        )
                        with gr.Row():
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
                        with gr.Row():
                            i2t_btn = gr.Button("Retrieve Matching Captions", variant="primary", scale=3)
                            i2t_clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                        i2t_probs = gr.Label(
                            num_top_classes=4,
                            label="Predicted Image Species",
                        )
                        i2t_status = gr.Markdown()

                i2t_table = gr.Dataframe(
                    label="Top-K Matching Captions",
                    interactive=False,
                    wrap=True,
                    max_height=380,
                    show_search="search",
                )

                if sample_images:
                    gr.Examples(
                        examples=sample_images,
                        example_labels=sample_labels,
                        inputs=[i2t_image],
                        label="Held-Out Test Set Samples (Click to Load)",
                    )

                i2t_inputs = [i2t_image, i2t_topk, i2t_split, i2t_species]
                i2t_outputs = [i2t_probs, i2t_status, i2t_table]

                i2t_btn.click(fn=run_image_to_text, inputs=i2t_inputs, outputs=i2t_outputs)
                i2t_topk.release(fn=_refresh_i2t_on_filter, inputs=i2t_inputs, outputs=i2t_outputs)
                i2t_split.change(fn=_refresh_i2t_on_filter, inputs=i2t_inputs, outputs=i2t_outputs)
                i2t_species.change(fn=_refresh_i2t_on_filter, inputs=i2t_inputs, outputs=i2t_outputs)
                i2t_clear_btn.click(
                    fn=_clear_i2t,
                    inputs=None,
                    outputs=[i2t_image, i2t_topk, i2t_split, i2t_species] + i2t_outputs,
                )

            # ---------------------------------------------------------------
            # Tab 3: Cross-Modal Pair Matcher
            # ---------------------------------------------------------------
            with gr.Tab("Pair Similarity Scorer"):
                gr.Markdown(
                    "Directly measure cosine similarity in $\\mathbb{S}^{255}$ between any wildlife image "
                    "and any custom caption, and compare auxiliary species predictions side-by-side."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        pair_image = gr.Image(
                            type="pil",
                            value=default_sample_path,
                            label="Wildlife Image",
                            height=300,
                        )
                    with gr.Column(scale=7):
                        pair_caption = gr.Textbox(
                            label="Caption to Compare",
                            placeholder="e.g., a tiger walking through dense brush",
                            value="a tiger walking through dense brush",
                            lines=2,
                        )
                        with gr.Row():
                            pair_btn = gr.Button("Compute Cosine Similarity", variant="primary", scale=3)
                            pair_clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                        pair_meter = gr.HTML(_build_pair_meter_html())
                        pair_summary = gr.Markdown()

                with gr.Row():
                    pair_img_probs = gr.Label(num_top_classes=4, label="Image Species Prediction")
                    pair_txt_probs = gr.Label(num_top_classes=4, label="Text Species Prediction")

                if sample_images:
                    gr.Examples(
                        examples=sample_images,
                        example_labels=sample_labels,
                        inputs=[pair_image],
                        label="Sample Test Images (Select to Load)",
                    )

                pair_outputs = [pair_meter, pair_summary, pair_img_probs, pair_txt_probs]
                pair_btn.click(fn=_run_pair_scorer_ui, inputs=[pair_image, pair_caption], outputs=pair_outputs)
                pair_caption.submit(fn=_run_pair_scorer_ui, inputs=[pair_image, pair_caption], outputs=pair_outputs)
                pair_clear_btn.click(
                    fn=_clear_pair,
                    inputs=None,
                    outputs=[pair_image, pair_caption] + pair_outputs,
                )

            # ---------------------------------------------------------------
            # Tab 4: Evaluation Metrics & Training Analytics
            # ---------------------------------------------------------------
            with gr.Tab("Evaluation & Analytics"):
                gr.Markdown(
                    "Live retrieval benchmark computing **Recall@1**, **Recall@5**, **Recall@10**, "
                    "**Mean Reciprocal Rank (MRR)**, and **Cosine Separation Margin**, alongside Member 3 training diagnostics."
                )
                with gr.Row():
                    eval_split = gr.Radio(
                        choices=["test", "validation", "train", "all"],
                        value="test",
                        label="Evaluation Split (Test = 30 Held-Out Samples)",
                        scale=3,
                    )
                    eval_btn = gr.Button("Run Evaluation Suite", variant="primary", scale=1)

                eval_kpis = gr.HTML()
                eval_headline = gr.Markdown()

                with gr.Row():
                    eval_overall_df = gr.Dataframe(
                        label="Overall Retrieval Metrics",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Row():
                    eval_species_df = gr.Dataframe(
                        label="Per-Species Breakdown",
                        interactive=False,
                        wrap=True,
                    )

                with gr.Accordion("Member 3 Training Curves & Test Confusion Matrices", open=True):
                    gr.HTML(
                        '<p class="wm-section-note">'
                        "Training convergence curves and held-out test set confusion matrices from Member 3 (Version 3)."
                        "</p>"
                    )
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

                eval_outputs = [eval_kpis, eval_headline, eval_overall_df, eval_species_df]
                eval_btn.click(fn=_run_evaluation_dashboard, inputs=[eval_split], outputs=eval_outputs)
                eval_split.change(fn=_run_evaluation_dashboard, inputs=[eval_split], outputs=eval_outputs)

        # Populate all tabs on initial page load so users never see blank panels
        demo.load(fn=run_text_to_image, inputs=t2i_inputs, outputs=t2i_outputs)
        demo.load(fn=run_image_to_text, inputs=i2t_inputs, outputs=i2t_outputs)
        demo.load(fn=_run_pair_scorer_ui, inputs=[pair_image, pair_caption], outputs=pair_outputs)
        demo.load(fn=_run_evaluation_dashboard, inputs=[eval_split], outputs=eval_outputs)

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
        theme=APP_THEME,
        css=CUSTOM_CSS,
        allowed_paths=[str(PROJECT_ROOT)],
    )


if __name__ == "__main__":
    main()
