"""Make six real Groq calls over held-out documents and save auditable outputs.

Run only when GROQ_API_KEY is configured (or present in faq-bot/.env). This is a
small functional check, not a statistical answer-accuracy estimate.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faq-bot"))
from dotenv import load_dotenv  # noqa: E402
from evaluation.realistic_benchmark import quote_in_text  # noqa: E402
from evaluation.realistic_corpus import HELDOUT  # noqa: E402
from model import answer_question  # noqa: E402
from rag import RAGIndex, bound_sources, evidence_sufficient  # noqa: E402

CASES = [
    ("east_library.txt", "How long can someone keep an adult print book from East Borough?",
     ["Adult print books are due after 21 calendar days."], [r"\b21\b"], False),
    ("bay_ferry.txt", "How late can I cancel a Bay Ferry booking for a cash refund?",
     ["A passenger may cancel a standard booking up to two hours before sailing and receive the fare less a six-dollar cancellation fee."],
     [r"\b(two|2)\s+hours?\b"], False),
    ("pinecrest_housing.txt", "What is Pinecrest's monthly ceiling on overnight guest stays?",
     ["An overnight visitor may stay for up to three consecutive nights, but no more than six nights in one calendar month."],
     [r"\b(six|6)\b", r"month"], False),
    ("museum_care.txt", "What humidity range does Meridian target for paintings?",
     ["The painting gallery targets relative humidity between 45 and 55 percent."],
     [r"\b45\b", r"\b55\b"], False),
    ("river_monitoring.txt", "What does the Riverbend team do when its oxygen meter fails calibration?",
     ["If a calibration check fails, staff mark the reading invalid, use a backup meter if available, and record the failure in the field log."],
     [r"invalid", r"backup", r"(log|record)"], False),
    (None, "What is the maximum security deposit at Pinecrest housing?", [],
     [r"(not\s+(specified|stated|provided)|does\s+not\s+(say|state|specify)|cannot\s+determine|don.t\s+know)"], True),
]


def citation_ids(answer):
    patterns = (r"【\s*(\d+)\s*】", r"\[\s*(?:Source\s+)?(\d+)\s*\]",
                r"\bSource\s+(\d+)\b")
    return {int(number) for pattern in patterns for number in re.findall(pattern, answer, re.I)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "grounding_results.json")
    args = parser.parse_args()
    load_dotenv(ROOT / "faq-bot" / ".env")
    index = RAGIndex(chunk_size=512, chunk_overlap=102, chunk_strategy="fixed",
                     cache_dir=ROOT / "evaluation" / ".cache")
    index.build_from_documents([{"source": doc["source"], "text": doc["text"]} for doc in HELDOUT])
    results = []
    for source, question, evidence, checks, unanswerable in CASES:
        retrieved = index.retrieve(question, top_k=5, mode="sparse", rerank=False)
        sources = bound_sources(retrieved, 2600)
        start = time.perf_counter()
        if not evidence_sufficient(question, sources):
            answer = "Insufficient evidence in the supplied context."
            called_groq = False
        else:
            answer = answer_question(question, sources)
            called_groq = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        citations = citation_ids(answer)
        source_ids = {chunk["chunk_id"] for chunk in sources}
        supporting_ids = {chunk["chunk_id"] for chunk in sources
                          if chunk["source"] == source and
                          all(quote_in_text(quote, chunk["text"]) for quote in evidence)}
        results.append({"question": question, "expected_source": source,
                        "unanswerable": unanswerable, "answer": answer,
                        "called_groq": called_groq, "groq_latency_ms": elapsed_ms,
                        "retrieved_source_ids": sorted(source_ids),
                        "supporting_source_ids": sorted(supporting_ids),
                        "citation_ids": sorted(citations),
                        "citations_valid": citations <= source_ids,
                        "cites_supporting_source": bool(citations & supporting_ids) if evidence else None,
                        "answer_check": all(re.search(pattern, answer, re.I) for pattern in checks)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
