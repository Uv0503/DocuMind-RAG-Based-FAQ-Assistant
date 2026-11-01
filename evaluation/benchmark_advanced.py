"""Reproducible synthetic retrieval and FAISS experiment (no LLM calls).

Run: python evaluation/benchmark_advanced.py --output evaluation/advanced_results.json
Use --smoke for a small pipeline check. The test split is evaluated once after
configuration selection on development data.
"""
import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "faq-bot"))
from rag import RAGIndex, faiss  # noqa: E402

SEED = 20261009
DOMAINS = {
    "astronomy": ("orbital survey", "spectral archive", "observatory calibration"),
    "botany": ("seed bank", "germination trial", "forest transect"),
    "logistics": ("freight corridor", "warehouse audit", "delivery manifest"),
    "music": ("rehearsal plan", "instrument repair", "concert archive"),
    "hydrology": ("river gauge", "watershed sampling", "floodplain review"),
    "archaeology": ("site excavation", "ceramic catalog", "stratigraphic survey"),
    "marine": ("reef survey", "coastal transect", "vessel log"),
    "agriculture": ("soil trial", "irrigation schedule", "harvest ledger"),
}

def narrative(domain, subject, code, rng):
    """Long-form operational context with document-specific details and distractors."""
    paragraphs = []
    for cycle in range(7):
        station = 100 + rng.randrange(800)
        sample = 20 + rng.randrange(180)
        paragraphs.append(
            f"During review cycle {cycle + 1}, the {domain} team examined {subject} record {code} "
            f"at station {station}. The field log contains {sample} observations and records the "
            f"operator's method, the date, and the instrument status for this specific survey. "
            f"The supervisor compared record {code} with its previous cycle and noted that "
            f"a change in sampling conditions did not alter the approved schedule. "
            f"A second reviewer checked the chronology and units for station {station} before "
            f"releasing the {domain} report. The team retained the original readings so a later "
            f"auditor could distinguish the accepted decision from earlier drafts."
        )
    return "\n\n".join(paragraphs)


def corpus(smoke=False):
    """Create facts first; evidence identifiers are labels independent of retrieval."""
    rng = random.Random(SEED)
    documents, queries = [], []
    per_domain = 3 if smoke else 20
    sections = 6 if smoke else 10
    for domain, subjects in DOMAINS.items():
        for document_number in range(per_domain):
            source = f"{domain}-{document_number:03d}.txt"
            paragraphs = []
            evidence_in_doc = []
            for section in range(sections):
                subject = subjects[section % len(subjects)]
                code = f"{domain[:3].upper()}{document_number:03d}{section:02d}"
                value = 15 + rng.randrange(170)
                revision = value + 7
                marker = f"EV{code}"
                heading = f"# {subject.title()} {section + 1}"
                fact = (f"Evidence {marker}: For {subject} record {code}, the approved review interval "
                        f"is {value} days. A draft proposed {revision} days, but that proposal was rejected. ")
                domain_context = (f"This {domain} file concerns {subject} record {code}. "
                                  f"Its review history belongs to the {domain} unit. ")
                paragraphs.append(heading + "\n" + narrative(domain, subject, code, rng) +
                                  "\n\n" + domain_context + fact)
                evidence_in_doc.append((marker, value, code))
                if document_number in {0, 4, 8, 12, 16} and section in {1, 5}:
                    queries.append({"id": marker, "split": "dev" if domain in list(DOMAINS)[:4] else "test",
                                    "question": (f"How often must {subject} record {code} be reviewed after the draft was rejected?"
                                                 if section == 5 else
                                                 f"What approved review interval applies to {subject} record {code}?"),
                                    "evidence": [marker], "answer": f"{value} days", "source": source,
                                    "kind": "paraphrase"})
                if document_number in {0, 4, 8, 12, 16} and section == 5:
                    earlier = evidence_in_doc[1]
                    queries.append({"id": f"HOP-{marker}",
                                    "split": "dev" if domain in list(DOMAINS)[:4] else "test",
                                    "question": f"Compare approved review intervals for records {earlier[2]} and {code} in {domain}.",
                                    "evidence": [earlier[0], marker],
                                    "answer": f"{earlier[1]} and {value} days", "source": source,
                                    "kind": "multi_hop"})
            documents.append({"source": source, "domain": domain, "text": "\n\n".join(paragraphs)})
        queries.append({"id": f"NONE-{domain}", "split": "dev" if domain in list(DOMAINS)[:4] else "test",
                        "question": f"What is the approved interval for the nonexistent {domain} record ZZ999?",
                        "evidence": [], "answer": None, "source": None, "kind": "unanswerable"})
    # The source document and domain are disjoint across splits. Labels predate chunking.
    dev_questions = [set(q["question"].casefold().split()) for q in queries if q["split"] == "dev"]
    test_questions = [set(q["question"].casefold().split()) for q in queries if q["split"] == "test"]
    if any(len(a & b) / len(a | b) >= .85 for a in dev_questions for b in test_questions):
        raise ValueError("Near-duplicate questions crossed the development/test split")
    return documents, queries


def metrics(index, queries, mode, rerank=False, top_k=10):
    # Each method pays its own query-encoding cost; earlier methods cannot warm
    # the embedding cache for later methods in the timing comparison.
    if hasattr(index, "query_embeddings"):
        index.query_embeddings.clear()
    ranks, coverage, precisions, ndcgs, latencies, empty_retrieval = [], [], [], [], [], []
    for query in queries:
        start = time.perf_counter()
        found = index.retrieve(query["question"], top_k=top_k, min_score=.10,
                               mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - start) * 1000)
        if query["evidence"]:
            first_rank = {}
            relevant_positions = []
            for position, chunk in enumerate(found, 1):
                matched = [marker for marker in query["evidence"]
                           if chunk["source"] == query["source"] and marker in chunk["text"]]
                if matched:
                    relevant_positions.append(position)
                for marker in matched:
                    first_rank.setdefault(marker, position)
            # A multi-hop question is retrieved only after every required fact is present.
            complete_rank = max(first_rank.values()) if len(first_rank) == len(query["evidence"]) else None
            ranks.append(complete_rank)
            precisions.append(sum(r <= 5 for r in relevant_positions) / min(5, max(1, len(found))))
            dcg = sum(1 / np.log2(rank + 1) for rank in relevant_positions)
            if hasattr(index, "chunks"):
                relevant_total = sum(chunk["source"] == query["source"] and
                                     any(marker in chunk["text"] for marker in query["evidence"])
                                     for chunk in index.chunks)
            else:
                relevant_total = len(query["evidence"])
            ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(top_k, relevant_total) + 1))
            ndcgs.append(dcg / ideal if ideal else 0)
            coverage.append(int(complete_rank is not None and complete_rank <= 5))
        else:
            empty_retrieval.append(int(not found))
    ordered = sorted(latencies)
    quantile = lambda p: ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
    return {"queries": len(queries), "answerable": len(ranks),
            **{f"recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks) / len(ranks) if ranks else None
               for k in (1, 5, 10)},
            "mrr_at_10": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks) if ranks else None,
            "ndcg_at_10": statistics.mean(ndcgs) if ndcgs else None,
            "precision_at_5": statistics.mean(precisions) if precisions else None,
            "answer_coverage_at_5": statistics.mean(coverage) if coverage else None,
            "unanswerable_empty_retrieval_rate": statistics.mean(empty_retrieval) if empty_retrieval else None,
            "latency_p50_ms": quantile(.5), "latency_p95_ms": quantile(.95)}


def ann_experiment(embeddings, query_vectors):
    if faiss is None:
        return {"unavailable": "faiss-cpu is not installed"}
    dim, n = embeddings.shape[1], len(embeddings)
    exact = faiss.IndexFlatIP(dim)
    exact.add(embeddings)
    truth = exact.search(query_vectors, 10)[1]
    results = {}
    configurations = [("flat", None), ("hnsw", 32)]
    # FAISS recommends roughly 39 training vectors per centroid; cap small-dev IVF grids.
    upper_nlist = max(2, min(64, int(np.sqrt(n)), n // 39))
    configurations += [("ivf", nlist) for nlist in sorted({max(1, upper_nlist // 2), upper_nlist})]
    for family, parameter in configurations:
        start = time.perf_counter()
        if family == "flat":
            index = faiss.IndexFlatIP(dim)
        elif family == "hnsw":
            index = faiss.IndexHNSWFlat(dim, parameter, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, parameter, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        index.add(embeddings)
        build_ms = (time.perf_counter() - start) * 1000
        search_params = ([None] if family == "flat" else
                         ([32, 64] if family == "hnsw" else sorted({min(parameter, 4), min(parameter, 8)})))
        for search_param in search_params:
            if family == "hnsw":
                index.hnsw.efSearch = search_param
            elif family == "ivf":
                index.nprobe = search_param
            times, recall = [], []
            for vector, expected in zip(query_vectors, truth):
                start = time.perf_counter()
                actual = index.search(vector[None, :], 10)[1][0]
                times.append((time.perf_counter() - start) * 1000)
                recall.append(len(set(actual) & set(expected)) / 10)
            label = family if search_param is None else f"{family}-{parameter}-{search_param}"
            results[label] = {"ann_recall_at_10": statistics.mean(recall),
                              "latency_p50_ms": statistics.median(times),
                              "latency_p95_ms": sorted(times)[int(.95 * (len(times) - 1))],
                              "build_ms": build_ms,
                              "serialized_bytes": len(faiss.serialize_index(index))}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "advanced_results.json")
    args = parser.parse_args()
    np.random.seed(SEED)
    if faiss is not None:
        faiss.omp_set_num_threads(1)
    RAGIndex()._load_model()  # Exclude model initialization/download from index-build timing.
    documents, queries = corpus(args.smoke)
    split = lambda name: ([{k: v for k, v in d.items() if k != "domain"} for d in documents
                           if (d["domain"] in list(DOMAINS)[:4]) == (name == "dev")],
                          [q for q in queries if q["split"] == name])
    dev_docs, dev_queries = split("dev")
    test_docs, test_queries = split("test")
    configs = [(size, round(size * overlap), "fixed")
               for size in (256, 512, 1024) for overlap in (.1, .2)]
    configs += [(512, round(512 * overlap), "structure") for overlap in (.1, .2)]
    if args.smoke:
        configs = [(256, 26, "structure")]
    tuning = {}
    dev_subset = [doc for doc in dev_docs if int(doc["source"].split("-")[1].split(".")[0]) in {0, 4, 8}]
    subset_sources = {doc["source"] for doc in dev_subset}
    for size, overlap, strategy in configs:
        index = RAGIndex(chunk_size=size, chunk_overlap=overlap,
                         chunk_strategy=strategy, cache_dir=ROOT / "evaluation" / ".cache")
        start = time.perf_counter()
        index.build_from_documents(dev_subset if not args.smoke else dev_docs)
        if index.embeddings is None:
            raise RuntimeError("Dense model unavailable; benchmark cannot compare methods")
        subset = [q for q in dev_queries if q["source"] in subset_sources]
        if args.smoke:
            subset = dev_queries
        tuning[f"{size}-{overlap}-{strategy}"] = {
            "chunks": len(index.chunks), "indexing_ms": (time.perf_counter() - start) * 1000,
            "dense": metrics(index, subset, "dense", rerank=False)}
    selected = max(tuning, key=lambda key: (tuning[key]["dense"]["mrr_at_10"],
                     tuning[key]["dense"]["recall_at_5"], -tuning[key]["indexing_ms"]))
    size, overlap, strategy = selected.split("-", 2)
    output = {"seed": SEED, "synthetic": True, "documents": len(documents),
              "development_documents": len(dev_docs), "test_documents": len(test_docs),
              "selected_chunking": selected, "tuning": tuning,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "faiss": getattr(faiss, "__version__", None),
                           "sentence_transformers": __import__("sentence_transformers").__version__,
                           "torch": __import__("torch").__version__},
              "embedding_model": RAGIndex().model_name,
              "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                           "faiss_threads": faiss.omp_get_max_threads() if faiss is not None else None},
              "test": {}}
    # Select ANN on development only. Require near-exact recall and a material speedup;
    # small document collections generally favor the exact index.
    dev_index = RAGIndex(chunk_size=int(size), chunk_overlap=int(overlap),
                         chunk_strategy=strategy, cache_dir=ROOT / "evaluation" / ".cache")
    dev_index.build_from_documents(dev_subset)
    dev_questions = [q for q in dev_queries if q["source"] in subset_sources]
    output["method_development"] = {}
    for method, use_reranker in (("sparse", False), ("dense", False),
                                 ("hybrid", False), ("hybrid", True)):
        label = "hybrid_rerank" if use_reranker else method
        if use_reranker:
            dev_index._load_reranker()
        output["method_development"][label] = metrics(dev_index, dev_questions, method, use_reranker)
    output["selected_method"] = max(output["method_development"], key=lambda name: (
        output["method_development"][name]["answer_coverage_at_5"],
        output["method_development"][name]["mrr_at_10"],
        -output["method_development"][name]["latency_p95_ms"]))
    dev_vectors = np.asarray(dev_index._load_model().encode(
        [q["question"] for q in dev_questions],
        convert_to_numpy=True, normalize_embeddings=True), dtype="float32")
    output["ann_development"] = ann_experiment(dev_index.embeddings, dev_vectors)
    if "flat" in output["ann_development"]:
        flat_latency = output["ann_development"]["flat"]["latency_p50_ms"]
        eligible = [(name, item) for name, item in output["ann_development"].items()
                    if item["ann_recall_at_10"] >= .95 and
                    item["latency_p50_ms"] <= .8 * flat_latency]
        output["selected_ann"] = min(eligible, key=lambda pair: pair[1]["latency_p50_ms"])[0] if eligible else "flat"
    else:
        output["selected_ann"] = "numpy_exact"
    # The untouched test set is indexed and evaluated only after the dev selection.
    test_index = RAGIndex(chunk_size=int(size), chunk_overlap=int(overlap),
                          chunk_strategy=strategy, cache_dir=ROOT / "evaluation" / ".cache")
    start = time.perf_counter()
    test_index.build_from_documents(test_docs)
    output["test_indexing_ms"] = (time.perf_counter() - start) * 1000
    output["test_chunks"] = len(test_index.chunks)
    baseline_index = RAGIndex(chunk_size=1000, chunk_overlap=150,
                              chunk_strategy="legacy_char", cache_dir=ROOT / "evaluation" / ".cache")
    baseline_index.build_from_documents(test_docs)
    output["legacy_baseline_chunks"] = len(baseline_index.chunks)
    output["test"]["legacy_baseline"] = metrics(baseline_index, test_queries, "baseline", False)
    for mode, rerank in (("sparse", False), ("dense", False),
                         ("hybrid", False), ("hybrid", True)):
        label = "hybrid_rerank" if rerank else mode
        if rerank:
            test_index._load_reranker()  # Benchmark must not mislabel a fallback as reranking.
        output["test"][label] = metrics(test_index, test_queries, mode, rerank)
    model = test_index._load_model()
    vectors = np.asarray(model.encode([q["question"] for q in test_queries],
                                      convert_to_numpy=True, normalize_embeddings=True), dtype="float32")
    output["ann_test"] = ann_experiment(test_index.embeddings, vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selected_chunking": selected, "selected_method": output["selected_method"],
                      "selected_ann": output["selected_ann"], "test_chunks": output["test_chunks"],
                      "test": output["test"], "ann_test": output["ann_test"]}, indent=2))


if __name__ == "__main__":
    main()
