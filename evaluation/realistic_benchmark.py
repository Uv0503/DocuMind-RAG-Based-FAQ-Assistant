"""Evidence-labeled document QA benchmark with disjoint development/test topics.

Run `python evaluation/realistic_benchmark.py --dev-only` for tuning, then run
without that flag once for the final held-out method comparison. No LLM calls.
"""
import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "faq-bot"))
from rag import RAGIndex, faiss  # noqa: E402
from evaluation.realistic_corpus import HELDOUT, UNANSWERABLE  # noqa: E402

SEED = 20261009
TOKEN_RE = re.compile(r"[a-z0-9]+")
CONFIGS = {
    "bm25": ("sparse", False, 1.0, 20, 512),
    "dense": ("dense", False, 1.0, 20, 512),
    "hybrid_rrf": ("hybrid", False, 1.0, 20, 512),
    "hybrid_sparse2": ("hybrid", False, 2.0, 20, 512),
    "hybrid_rerank_full": ("hybrid", True, 1.0, 20, 512),
    "hybrid_rerank_fast": ("hybrid", True, 1.0, 8, 256),
    "hybrid_sparse2_rerank_fast": ("hybrid", True, 2.0, 8, 256),
}


def tokens(text):
    return TOKEN_RE.findall(text.casefold())


def quote_in_text(quote, text):
    wanted, actual = tokens(quote), tokens(text)
    if not wanted:
        return False
    if " ".join(wanted) in " ".join(actual):
        return True
    expected, found = Counter(wanted), Counter(actual)
    return sum(min(count, found[word]) for word, count in expected.items()) / len(wanted) >= .8


def load_cases():
    dataset = json.loads((ROOT / "evaluation" / "eval_dataset.json").read_text(encoding="utf-8"))
    dev_docs = []
    for path in sorted((ROOT / "evaluation" / "documents").glob("*.txt")):
        dev_docs.append({"source": path.name, "text": path.read_text(encoding="utf-8"),
                         "topic": "company_policy"})
    # Existing unrelated demo documents raise candidate pressure without adding labels.
    for filename in ("abs.txt", "fileInput (2).txt", "random.txt"):
        path = ROOT / "DemoDocs" / filename
        dev_docs.append({"source": "demo_" + filename, "text": path.read_text(encoding="utf-8"),
                         "topic": "software_demo"})
    dev_cases = [{"id": q["id"], "question": q["question"], "source": q["document"],
                  "evidence": [q["relevant_text"]], "answer": q["expected_answer"]}
                 for q in dataset["queries"]]
    dev_cases += [
        {"id": "dev-unknown-pension", "question": "What percentage does Northstar contribute to pensions?",
         "source": None, "evidence": [], "answer": None},
        {"id": "dev-unknown-childcare", "question": "Does Northstar reimburse childcare during business travel?",
         "source": None, "evidence": [], "answer": None},
    ]
    test_docs = [{"source": item["source"], "text": item["text"], "topic": item["topic"]}
                 for item in HELDOUT]
    test_cases = []
    for item in HELDOUT:
        for number, (question, evidence, answer) in enumerate(item["questions"], 1):
            test_cases.append({"id": f"{item['source']}:{number}", "question": question,
                               "source": item["source"], "evidence": evidence, "answer": answer})
    test_cases += [{"id": f"unanswerable:{number}", "question": question,
                    "source": None, "evidence": [], "answer": None}
                   for number, question in enumerate(UNANSWERABLE, 1)]

    dev_sources = {doc["source"] for doc in dev_docs}
    test_sources = {doc["source"] for doc in test_docs}
    if dev_sources & test_sources:
        raise ValueError("source leakage across splits")
    if {doc["topic"] for doc in dev_docs} & {doc["topic"] for doc in test_docs}:
        raise ValueError("topic leakage across splits")
    for docs, cases in ((dev_docs, dev_cases), (test_docs, test_cases)):
        by_source = {doc["source"]: doc["text"] for doc in docs}
        for case in cases:
            if case["evidence"]:
                if case["source"] not in by_source or not all(
                        " ".join(tokens(quote)) in " ".join(tokens(by_source[case["source"]]))
                        for quote in case["evidence"]):
                    raise ValueError(f"unverified evidence label: {case['id']}")
    dev_questions = [set(tokens(q["question"])) for q in dev_cases]
    test_questions = [set(tokens(q["question"])) for q in test_cases]
    if any(len(a & b) / len(a | b) >= .85 for a in dev_questions for b in test_questions):
        raise ValueError("near-duplicate question leakage across splits")
    return dev_docs, dev_cases, test_docs, test_cases


def score_ranking(found, case):
    if not case["evidence"]:
        return {"complete_rank": None, "relevant_ranks": [], "quote_ranks": []}
    quote_ranks = []
    for quote in case["evidence"]:
        quote_ranks.append(next((rank for rank, chunk in enumerate(found, 1)
                                 if chunk["source"] == case["source"] and
                                 quote_in_text(quote, chunk["text"])), None))
    relevant_ranks = [rank for rank, chunk in enumerate(found, 1)
                      if chunk["source"] == case["source"] and
                      any(quote_in_text(quote, chunk["text"]) for quote in case["evidence"])]
    return {"complete_rank": max(quote_ranks) if all(quote_ranks) else None,
            "relevant_ranks": relevant_ranks, "quote_ranks": quote_ranks}


def build_index(documents, chunk_size=128):
    index = RAGIndex(chunk_size=chunk_size, chunk_overlap=round(chunk_size * .2),
                     chunk_strategy="fixed", cache_dir=ROOT / "evaluation" / ".cache")
    index.build_from_documents([{"source": doc["source"], "text": doc["text"]}
                                for doc in documents])
    if index.embeddings is None:
        raise RuntimeError("dense embeddings unavailable; comparison cannot continue")
    return index


def evaluate(index, cases, config):
    mode, rerank, weight, top_n, max_length = config
    index.sparse_weight = weight
    index.rerank_top_n = top_n
    index.reranker_max_length = max_length
    index.reranker = None
    if rerank:
        index._load_reranker()  # Exclude model loading from per-query latency.
    index.query_embeddings.clear()
    ranks, ndcgs, latencies, unanswerable_empty = [], [], [], []
    for case in cases:
        start = time.perf_counter()
        found = index.retrieve(case["question"], top_k=10, mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - start) * 1000)
        if not case["evidence"]:
            unanswerable_empty.append(int(not found))
            continue
        scored = score_ranking(found, case)
        ranks.append(scored["complete_rank"])
        relevant_total = sum(chunk["source"] == case["source"] and
                             any(quote_in_text(quote, chunk["text"]) for quote in case["evidence"])
                             for chunk in index.chunks)
        if relevant_total == 0:
            raise ValueError(f"evidence not mapped to any chunk: {case['id']}")
        dcg = sum(1 / np.log2(rank + 1) for rank in scored["relevant_ranks"])
        ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(10, relevant_total) + 1))
        ndcgs.append(dcg / ideal)
    sorted_latency = sorted(latencies)
    percentile = lambda p: sorted_latency[min(len(sorted_latency) - 1, int(p * (len(sorted_latency) - 1)))]
    return {"answerable": len(ranks), "unanswerable": len(unanswerable_empty),
            "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
            "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
            "mrr_at_10": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
            "ndcg_at_10": statistics.mean(ndcgs),
            "unanswerable_empty_rate": statistics.mean(unanswerable_empty) if unanswerable_empty else None,
            "latency_p50_ms": percentile(.5), "latency_p95_ms": percentile(.95)}


def choose(dev):
    best_recall = max(result["recall_at_5"] for result in dev.values())
    eligible = {name: result for name, result in dev.items()
                if result["recall_at_5"] >= best_recall - 1e-9}
    best_mrr = max(result["mrr_at_10"] for result in eligible.values())
    near_best = {name: result for name, result in eligible.items()
                 if result["mrr_at_10"] >= best_mrr - .015}
    return min(near_best, key=lambda name: near_best[name]["latency_p95_ms"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "realistic_results.json")
    args = parser.parse_args()
    np.random.seed(SEED)
    if faiss is not None:
        faiss.omp_set_num_threads(1)
    import torch
    torch.set_num_threads(2)
    dev_docs, dev_cases, test_docs, test_cases = load_cases()
    dev_index = build_index(dev_docs)
    dev = {name: evaluate(dev_index, dev_cases, config) for name, config in CONFIGS.items()}
    selected = choose(dev)
    output = {"seed": SEED, "chunk_size": 128, "chunk_overlap": 26,
              "development_sources": [doc["source"] for doc in dev_docs],
              "development_chunks": len(dev_index.chunks), "development": dev,
              "selection_rule": "best dev Recall@5; within 0.015 of best MRR@10; then minimum p95 latency",
              "selected_method": selected,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "faiss": getattr(faiss, "__version__", None),
                           "torch": torch.__version__,
                           "sentence_transformers": __import__("sentence_transformers").__version__},
              "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                           "faiss_threads": faiss.omp_get_max_threads() if faiss else None}}
    if not args.dev_only:
        test_index = build_index(test_docs)
        output["test_sources"] = [doc["source"] for doc in test_docs]
        output["test_chunks"] = len(test_index.chunks)
        output["test"] = {name: evaluate(test_index, test_cases, config)
                          for name, config in CONFIGS.items()}
        baseline_index = build_index(test_docs, chunk_size=512)
        output["previous_production"] = evaluate(baseline_index, test_cases, CONFIGS["bm25"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selected_method": selected, "development": dev,
                      "test": output.get("test"), "previous_production": output.get("previous_production")}, indent=2))


if __name__ == "__main__":
    main()
