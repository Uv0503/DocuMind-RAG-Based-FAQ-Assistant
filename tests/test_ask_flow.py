import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "faq-bot"))
import main


class AskFlowTests(unittest.TestCase):
    def test_missing_identifier_abstains_without_groq_call(self):
        sources = [{"chunk_id": 1, "source": "policy.txt", "heading": "Rules",
                    "text": "Record ORC123 is reviewed every seven days.", "score": 1.0}]
        with patch.object(main.rag_index, "retrieve", return_value=sources), \
             patch.object(main, "log_interaction", new_callable=AsyncMock) as logged, \
             patch.object(main, "answer_question") as groq:
            response = asyncio.run(main.ask(main.AskRequest(question="What applies to ZZ999?", debug=True),
                                            user={"id": 1}))
        self.assertEqual(response["answer"], main.FALLBACK_ANSWER)
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["retrieval_debug"]["reason"], "missing_evidence_or_record_identifier")
        logged.assert_awaited_once()
        groq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
