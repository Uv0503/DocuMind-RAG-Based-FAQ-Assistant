import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.validate_grounding import citation_ids


class CitationParsingTests(unittest.TestCase):
    def test_recognizes_model_source_citation_formats(self):
        self.assertEqual(citation_ids('See 【5】 and [Source 2].'), {2, 5})
        self.assertEqual(citation_ids('No citation here: 21 days.'), set())


if __name__ == '__main__':
    unittest.main()
