import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "faq-bot"))
from rag import RAGIndex, bound_sources, evidence_sufficient


class RetrievalTests(unittest.TestCase):
    def test_filtering_refills_from_lower_ranked_valid_candidates(self):
        index = RAGIndex(candidate_k=2)
        index.chunks = [{"chunk_id": n + 1, "source": f"{n}.txt", "heading": "", "text": f"passage {n}"}
                        for n in range(3)]
        index.is_built = True
        index._sparse = Mock(return_value=[(1, 5.0), (2, 4.0)])
        index._dense = Mock(return_value=[(0, .05)])
        found = index.retrieve("passage", top_k=2, min_score=.1, mode="hybrid", rerank=False)
        self.assertEqual([chunk["source"] for chunk in found], ["1.txt", "2.txt"])

    def test_weighted_rrf_can_preserve_strong_sparse_rank(self):
        index = RAGIndex()
        index.chunks = [{"chunk_id": n + 1, "source": f"{n}.txt", "heading": "", "text": f"passage {n}"}
                        for n in range(3)]
        index.is_built = True
        index._sparse = Mock(return_value=[(0, 9.0), (1, 8.0), (2, 7.0)])
        index._dense = Mock(return_value=[(1, .9), (2, .8), (0, .1)])
        index.sparse_weight = 3.0
        self.assertEqual(index.retrieve("passage", top_k=1, mode="hybrid", rerank=False)[0]["source"], "0.txt")

    def test_reranker_scores_only_configured_leading_candidates(self):
        index = RAGIndex()
        index.chunks = [{"chunk_id": n + 1, "source": f"{n}.txt", "heading": "", "text": f"passage {n}"}
                        for n in range(4)]
        index.is_built = True
        index._sparse = Mock(return_value=[(n, 4.0 - n) for n in range(4)])
        index._dense = Mock(return_value=[])
        index.rerank_top_n = 2
        index.reranker = Mock()
        index.reranker.predict.return_value = np.array([0.0, 1.0])
        result = index.retrieve("passage", top_k=4, mode="sparse", rerank=True)
        self.assertEqual([r["source"] for r in result], ["1.txt", "0.txt", "2.txt", "3.txt"])
        self.assertEqual(len(index.reranker.predict.call_args.args[0]), 2)

    def test_unknown_record_identifier_requires_abstention(self):
        sources = [{"text": "The approved interval for record ORC123 is seven days."}]
        self.assertTrue(evidence_sufficient("What is the interval for ORC123?", sources))
        self.assertFalse(evidence_sufficient("What is the interval for ZZ999?", sources))

    def test_defaults_follow_development_selection(self):
        with patch.dict("os.environ", {}, clear=True):
            index = RAGIndex()
            self.assertEqual((index.chunk_size, index.chunk_overlap, index.chunk_strategy), (512, 51, "fixed"))
            self.assertEqual((index.rerank_top_n, index.reranker_max_length), (8, 256))

    def test_query_embedding_is_reused_across_retrieval_modes(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=30, chunk_overlap=0, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.ones((1, 2), dtype="float32")
            index.build_from_documents([{"source": "a.txt", "text": "Orchid seven days."}])
            index.retrieve("Orchid days", mode="dense", rerank=False)
            index.retrieve("Orchid days", mode="hybrid", rerank=False)
            self.assertEqual(index.model.encode.call_count, 2)  # document once, question once

    def test_context_budget_counts_tokens_and_keeps_metadata(self):
        chunks = [{"source": "a.txt", "heading": "Policy", "text": "alpha, beta gamma delta"},
                  {"source": "b.txt", "text": "other evidence"}]
        bounded = bound_sources(chunks, 3)
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0]["text"], "alpha, beta")
        self.assertEqual(bounded[0]["heading"], "Policy")

    def test_fixed_chunks_keep_nearest_heading(self):
        index = RAGIndex(chunk_size=5, chunk_overlap=0, chunk_strategy="fixed")
        chunks = index._split_with_headings("# Alpha\nred blue green\n# Beta\nyellow purple orange")
        self.assertEqual(chunks[0][0], "Alpha")
        self.assertEqual(chunks[-1][0], "Beta")

    def test_embedding_model_environment_setting_is_used(self):
        with patch.dict("os.environ", {"EMBEDDING_MODEL": "custom/model"}):
            self.assertEqual(RAGIndex().model_name, "custom/model")

    def test_dense_query_failure_falls_back_to_bm25(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=30, chunk_overlap=0, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.ones((1, 2), dtype="float32")
            index.build_from_documents([{"source": "a.txt", "text": "# Policy\nOrchid grants seven days."}])
            index.model.encode.side_effect = RuntimeError("model unavailable")
            found = index.retrieve("Orchid days", mode="hybrid", rerank=False)
            self.assertEqual(found[0]["source"], "a.txt")
            self.assertIsNone(found[0]["dense_score"])

    def test_structure_chunks_preserve_headings_and_document_sources(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=10, chunk_overlap=1, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.ones((2, 2), dtype="float32")
            count = index.build_from_documents([
                {"source": "a.txt", "text": "# Apples\nred green blue"},
                {"source": "b.txt", "text": "# Pears\nyellow green blue"},
            ])
            self.assertEqual(count, 2)
            self.assertEqual([c["heading"] for c in index.chunks], ["Apples", "Pears"])
            self.assertEqual([c["source"] for c in index.chunks], ["a.txt", "b.txt"])

    def test_bm25_prefers_specific_term_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=30, chunk_overlap=0, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.ones((2, 2), dtype="float32")
            count = index.build_from_documents([
                {"source": "first.txt", "text": "# Rules\nOrchid permits ten days."},
                {"source": "duplicate.txt", "text": "# Rules\nOrchid permits ten days."},
                {"source": "other.txt", "text": "# Rules\nCedar permits five days."},
            ])
            self.assertEqual(count, 2)
            result = index.retrieve("Orchid days", mode="sparse", rerank=False)
            self.assertEqual(result[0]["source"], "first.txt")
            self.assertGreater(result[0]["bm25_score"], 0)

    def test_hybrid_fuses_and_reranks_without_dropping_metadata(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=30, chunk_overlap=0, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.array([[1, 0], [0, 1]], dtype="float32")
            index.build_from_documents([
                {"source": "a.txt", "text": "# One\nAlpha value."},
                {"source": "b.txt", "text": "# Two\nBeta value."},
            ])
            index.model.encode.return_value = np.array([[1, 0]], dtype="float32")
            index.reranker = Mock()
            index.reranker.predict.return_value = np.array([0.1, 0.9])
            result = index.retrieve("Alpha Beta", top_k=2, mode="hybrid", rerank=True)
            self.assertEqual({r["source"] for r in result}, {"a.txt", "b.txt"})
            self.assertEqual(result[0]["source"], "b.txt")
            self.assertIsNotNone(result[0]["rerank_score"])

    def test_nonexistent_explicit_record_abstains(self):
        with tempfile.TemporaryDirectory() as path:
            index = RAGIndex(chunk_size=30, chunk_overlap=0, cache_dir=Path(path))
            index.model = Mock()
            index.model.encode.return_value = np.array([[1, 0]], dtype="float32")
            index.build_from_documents([{"source": "a.txt", "text": "Approved interval for record AB1234 is ten days."}])
            self.assertEqual(index.retrieve("Approved interval for record ZZ9999?",
                                            mode="hybrid", rerank=False), [])


if __name__ == "__main__":
    unittest.main()
