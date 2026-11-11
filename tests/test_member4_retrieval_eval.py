"""
Unit and integration tests for Member 4 deliverables:
- retrieval.py (cosine similarity, gallery indexing, T2I/I2T retrieval, pair scoring)
- evaluation.py (Recall@1, Recall@5, Recall@10, MRR, split evaluation)
- app.py (Gradio Blocks construction and UI callback functions)
"""

# Import torch FIRST to avoid Windows OpenMP/MKL DLL initialization conflicts
import torch

import sys
import unittest
from pathlib import Path
from PIL import Image
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wildmatch_v3 import SPECIES_CLASSES
from retrieval import calculate_cosine_similarity, WildMatchRetriever
from evaluation import (
    compute_recall_at_k,
    compute_mrr,
    compute_retrieval_metrics,
    evaluate_retriever,
)
if HAS_GRADIO:
    import app as wildmatch_app


class TestMember4MetricsMath(unittest.TestCase):
    def test_cosine_similarity_invariants(self):
        q = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        sim = calculate_cosine_similarity(q, c)
        self.assertEqual(sim.shape, (2, 3))
        self.assertAlmostEqual(sim[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(sim[0, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(sim[0, 2].item(), -1.0, places=5)
        self.assertAlmostEqual(sim[1, 1].item(), 1.0, places=5)

    def test_recall_and_mrr_exact_values(self):
        # 3 queries, 4 candidates
        # Query 0: best rank of relevant (col 0) is rank 1 -> RR = 1.0
        # Query 1: best rank of relevant (col 2) is rank 2 -> RR = 0.5
        # Query 2: best rank of relevant (col 3) is rank 3 -> RR = 1/3
        sim = torch.tensor(
            [
                [0.9, 0.4, 0.2, 0.1],
                [0.8, 0.1, 0.6, 0.2],
                [0.7, 0.6, 0.1, 0.5],
            ]
        )
        rel = torch.tensor(
            [
                [True, False, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]
        )
        r1 = compute_recall_at_k(sim, rel, k=1)
        r2 = compute_recall_at_k(sim, rel, k=2)
        r5 = compute_recall_at_k(sim, rel, k=5)  # Clamped to 4 candidates
        mrr = compute_mrr(sim, rel)

        self.assertAlmostEqual(r1, 1.0 / 3.0, places=5)
        self.assertAlmostEqual(r2, 2.0 / 3.0, places=5)
        self.assertAlmostEqual(r5, 1.0, places=5)
        self.assertAlmostEqual(mrr, (1.0 + 0.5 + 1.0 / 3.0) / 3.0, places=5)

        bundle = compute_retrieval_metrics(sim, rel, ks=(1, 2, 4))
        self.assertAlmostEqual(bundle["recall@1"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(bundle["recall@2"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(bundle["recall@4"], 1.0, places=5)
        self.assertAlmostEqual(bundle["mrr"], mrr, places=5)


_TEST_CSV = PROJECT_ROOT / "dataset" / "splits" / "test.csv"
def _check_test_images() -> bool:
    if not _TEST_CSV.is_file():
        return False
    df = pd.read_csv(_TEST_CSV)
    return all((PROJECT_ROOT / p).is_file() for p in df["image_path"])

_HAS_IMAGES = _check_test_images()


@unittest.skipUnless(_HAS_IMAGES, "complete test split dataset images not available locally")
class TestMember4RetrievalAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = WildMatchRetriever(split="test")

    def test_test_split_indexed(self):
        self.assertEqual(len(self.retriever.records), 30)
        self.assertEqual(self.retriever.text_embeddings.shape, (30, 256))
        self.assertEqual(len(self.retriever.image_record_indices), 30)
        self.assertEqual(self.retriever.image_embeddings.shape, (30, 256))

    def test_text_to_image_topk_and_ranking(self):
        res = self.retriever.retrieve_images_by_text(
            "a tiger walking through tall grass", top_k=10
        )
        self.assertEqual(res["predicted_species"], "Tiger")
        self.assertEqual(len(res["results"]), 10)
        scores = [r["similarity"] for r in res["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res["results"][0]["species"], "Tiger")

    def test_text_to_image_species_filter(self):
        res = self.retriever.retrieve_images_by_text(
            "animal in forest", top_k=5, species_filter="Chital"
        )
        self.assertEqual(len(res["results"]), 5)
        self.assertTrue(all(r["species"] == "Chital" for r in res["results"]))

    def test_image_to_text_pil_and_path(self):
        sample_rec = self.retriever.records[0]
        res_path = self.retriever.retrieve_texts_by_image(sample_rec["image_path"], top_k=5)
        self.assertEqual(len(res_path["results"]), 5)
        scores = [r["similarity"] for r in res_path["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

        with Image.open(sample_rec["image_path"]) as pil_img:
            res_pil = self.retriever.retrieve_texts_by_image(pil_img, top_k=5)
        self.assertEqual(res_pil["predicted_species"], res_path["predicted_species"])

    def test_pair_scorer(self):
        tiger_rec = next(r for r in self.retriever.records if r["species"] == "Tiger")
        pair = self.retriever.score_image_text_pair(
            tiger_rec["image_path"], "a tiger resting near shallow water"
        )
        self.assertGreaterEqual(pair["cosine_similarity"], -1.0)
        self.assertLessEqual(pair["cosine_similarity"], 1.0)
        self.assertEqual(pair["text_predicted_species"], "Tiger")

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve_images_by_text("   ")
        with self.assertRaises(ValueError):
            self.retriever.score_image_text_pair(self.retriever.records[0]["image_path"], "")

    def test_evaluation_suite_on_test_split(self):
        ev = evaluate_retriever(self.retriever, split="test")
        m = ev["metrics"]
        self.assertEqual(ev["num_samples"], 30)
        self.assertAlmostEqual(m["t2i_species_r1"], 1.0, places=4)
        self.assertAlmostEqual(m["t2i_species_r5"], 1.0, places=4)
        self.assertAlmostEqual(m["t2i_species_mrr"], 1.0, places=4)
        self.assertGreaterEqual(m["i2t_species_r1"], 0.80)
        self.assertAlmostEqual(m["i2t_species_r5"], 0.90, places=5)
        self.assertGreater(m["species_mrr_average"], 0.85)
        self.assertGreater(ev["similarity_stats"]["species_separation_margin"], 0.4)
        for sp in SPECIES_CLASSES:
            self.assertIn(sp, ev["per_species"])


@unittest.skipUnless(HAS_GRADIO and _HAS_IMAGES, "gradio and complete test images required")
class TestMember4GradioApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = WildMatchRetriever(split="test")
        wildmatch_app._RETRIEVER = cls.retriever
    def test_gradio_blocks_and_callbacks(self):
        demo = wildmatch_app.build_app()
        self.assertIsInstance(demo, gr.Blocks)

        probs, status, gallery, df = wildmatch_app.run_text_to_image(
            "a lion walking across a dry field", 5, "test", "All"
        )
        self.assertEqual(len(probs), 4)
        self.assertIn("Lion", status)
        self.assertEqual(len(gallery), 5)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)

        retriever = wildmatch_app.get_retriever()
        sample_path = retriever.records[0]["image_path"]
        with Image.open(sample_path) as img:
            i2t_probs, i2t_status, i2t_df = wildmatch_app.run_image_to_text(
                img, 5, "test", "All"
            )
            self.assertEqual(len(i2t_probs), 4)
            self.assertEqual(len(i2t_df), 5)
            self.assertIn("Predicted Image Species", i2t_status)

            pair_md, p_img, p_txt = wildmatch_app.run_pair_scorer(
                img, "a tiger walking through the woods"
            )
            self.assertIn("Cosine Similarity", pair_md)
            self.assertEqual(len(p_img), 4)
            self.assertEqual(len(p_txt), 4)

        ev_md, ev_overall, ev_species = wildmatch_app.run_evaluation_ui("test")
        self.assertIn("Evaluation Summary", ev_md)
        self.assertEqual(len(ev_overall), 5)
        self.assertEqual(len(ev_species), 4)

    def test_ui_helpers_empty_states_and_escaping(self):
        # Empty-state guidance for T2I, I2T, and Pair Scorer
        t2i_probs, t2i_status, t2i_gal, t2i_df = wildmatch_app.run_text_to_image(
            "   ", 5, "all", "All"
        )
        self.assertEqual(t2i_probs, {})
        self.assertIn("Enter a Wildlife Description", t2i_status)
        self.assertEqual(t2i_gal, [])
        self.assertTrue(t2i_df.empty)

        i2t_probs, i2t_status, i2t_df = wildmatch_app.run_image_to_text(
            None, 5, "all", "All"
        )
        self.assertEqual(i2t_probs, {})
        self.assertIn("Upload or Select", i2t_status)
        self.assertTrue(i2t_df.empty)

        meter_empty, summary_empty, p_img, p_txt = wildmatch_app._run_pair_scorer_ui(
            None, "a tiger"
        )
        self.assertIn("wm-alert-empty", meter_empty)
        self.assertIn("upload or select", summary_empty.lower())
        self.assertEqual(p_img, {})
        self.assertEqual(p_txt, {})

        # HTML escaping in meter builder
        escaped_html = wildmatch_app._build_pair_meter_html(message="<script>alert(1)</script>")
        self.assertNotIn("<script>", escaped_html)
        self.assertIn("&lt;script&gt;", escaped_html)

        # Dashboard KPI cards & Pair meter populated path
        kpis_html, dash_md, dash_overall, dash_species = wildmatch_app._run_evaluation_dashboard("test")
        self.assertIn("wm-kpi-grid", kpis_html)
        self.assertIn("Evaluation Summary", dash_md)
        self.assertEqual(len(dash_overall), 5)
        self.assertEqual(len(dash_species), 4)

        retriever = wildmatch_app.get_retriever()
        sample_path = retriever.records[0]["image_path"]
        with Image.open(sample_path) as img:
            meter_ok, pair_md, img_p, txt_p = wildmatch_app._run_pair_scorer_ui(
                img, "a tiger walking through tall grass"
            )
            self.assertIn("wm-meter-card", meter_ok)
            self.assertIn("Cosine Similarity", pair_md)
            self.assertEqual(len(img_p), 4)
            self.assertEqual(len(txt_p), 4)

        # Clear handlers return expected tuple lengths
        self.assertEqual(len(wildmatch_app._clear_t2i()), 8)
        self.assertEqual(len(wildmatch_app._clear_i2t()), 7)
        self.assertEqual(len(wildmatch_app._clear_pair()), 6)


if __name__ == "__main__":
    unittest.main()
