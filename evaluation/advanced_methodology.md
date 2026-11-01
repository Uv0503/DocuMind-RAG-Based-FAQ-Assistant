# Advanced retrieval benchmark

Run from the repository root after installing `faq-bot/requirements.txt`:

```bash
python evaluation/benchmark_advanced.py --smoke --output /tmp/documind-smoke.json
python evaluation/benchmark_advanced.py --output evaluation/advanced_results.json
```

The generator uses seed `20261009` to create 160 long TXT documents in eight unrelated domains. At the 256-token fixed configuration this produces 5,440 chunks. Each fact is assigned an evidence ID and answer when its source text is generated; labels do not come from search results. Facts include rejected draft values as distractors, paraphrased questions, two-fact questions, and unanswerable questions. Development and test domains and source files do not overlap. Document-specific record codes and station details reduce duplicate passages; the shared writing template remains a limitation. The tuning subset samples development documents from all four development domains. Chunking and ANN settings are selected from development results before test indexing. No Groq calls are made.

The benchmark compares the original 1,000-character/150-character-overlap dense baseline with BM25, dense, RRF hybrid and RRF plus Cross-Encoder on the selected token chunks. It reports relevance Recall@1/5/10, MRR@10, nDCG@10, precision@5, evidence coverage@5, the empty retrieval rate for unanswerable questions, and retrieval latency. A two-fact question counts as recalled only when both evidence IDs appear within K results. The empty retrieval rate is **not** answer abstention. Evidence coverage is an extractive proxy, **not** a measured Groq answer-correctness score. The token splitter counts regex word/punctuation tokens; the embedding model may internally truncate long chunks, especially at 512 and 1024 tokens. Synthetic writing remains easier than unseen human text.

The FAISS experiment holds the selected normalized embedding matrix and query vectors constant. It compares IndexFlatIP, IndexHNSWFlat (M=32; efSearch 32/64) and IndexIVFFlat (two nlist values near sqrt(N); nprobe 4/8). ANN recall@10 is neighbor overlap against IndexFlatIP; it is separate from evidence recall. Serialized index bytes approximate index storage, not process RSS. Build time and p50/p95 per-query latency are recorded. The development selection requires at least 0.95 neighbor recall and a 20% median latency reduction over FlatIP; otherwise it keeps exact search. Test ANN measurements validate that choice without tuning it.

Embedding matrices are cached by model, chunk configuration and content under `evaluation/.cache/`. Remove that ignored directory to force a cold run. Times are machine dependent. The result JSON records Python, NumPy and FAISS versions and seed. A run fails if dense embeddings cannot load, rather than silently recording BM25 fallback as a dense result.
