import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.benchmark_advanced import metrics


class BenchmarkMetricTests(unittest.TestCase):
    def test_multi_hop_recall_requires_both_evidence_markers(self):
        class Index:
            def retrieve(self, *_args, **_kwargs):
                return [{"source": "a.txt", "text": "Evidence EV1"}]

        query = {"question": "compare", "source": "a.txt", "evidence": ["EV1", "EV2"]}
        result = metrics(Index(), [query], "sparse")
        self.assertEqual(result["recall_at_10"], 0)
        self.assertEqual(result["answer_coverage_at_5"], 0)

    def test_unanswerable_does_not_claim_answer_abstention(self):
        class Index:
            def retrieve(self, *_args, **_kwargs):
                return [{"source": "other.txt", "text": "distractor"}]

        query = {"question": "unknown", "source": None, "evidence": []}
        result = metrics(Index(), [query], "sparse")
        self.assertEqual(result["unanswerable_empty_retrieval_rate"], 0)
        self.assertIsNone(result["recall_at_10"])


if __name__ == "__main__":
    unittest.main()
