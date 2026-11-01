import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.realistic_benchmark import choose, load_cases, score_ranking


class RealisticBenchmarkTests(unittest.TestCase):
    def test_selection_prefers_recall_then_meaningful_mrr_gain(self):
        values = {
            'bm25': {'recall_at_5': .967, 'mrr_at_10': .91, 'latency_p95_ms': .1},
            'hybrid': {'recall_at_5': 1.0, 'mrr_at_10': .98, 'latency_p95_ms': 7.0},
            'reranked': {'recall_at_5': 1.0, 'mrr_at_10': .96, 'latency_p95_ms': 180.0},
        }
        self.assertEqual(choose(values), 'hybrid')

    def test_curated_evidence_is_verified_before_retrieval(self):
        dev_docs, dev_cases, test_docs, test_cases = load_cases()
        self.assertTrue(dev_docs and test_docs)
        self.assertGreaterEqual(len(test_cases), 25)
        self.assertFalse({d['source'] for d in dev_docs} & {d['source'] for d in test_docs})
        self.assertTrue(all(case['evidence'] for case in test_cases if case['answer'] is not None))

    def test_multi_hop_requires_every_labeled_passage(self):
        case = {'source': 'a.txt', 'evidence': ['first fact', 'second fact'], 'answer': 'both'}
        one = [{'source': 'a.txt', 'text': 'first fact'}]
        both = one + [{'source': 'a.txt', 'text': 'second fact'}]
        self.assertIsNone(score_ranking(one, case)['complete_rank'])
        self.assertEqual(score_ranking(both, case)['complete_rank'], 2)


if __name__ == '__main__':
    unittest.main()
