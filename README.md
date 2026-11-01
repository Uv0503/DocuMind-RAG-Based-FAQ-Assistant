# DocuMind - RAG-Based FAQ Assistant

**DocuMind** is a full-stack, document-grounded FAQ assistant. Upload a UTF-8 `.txt` document, ask a question, and receive an answer generated from the most relevant parts of that document -- along with the source chunks used to create it.

[View the project on GitHub](https://github.com/Uv0503/DocuMind-RAG-Based-FAQ-Assistant)

## Features

- Upload and index a `.txt` knowledge document
- Generate grounded answers with a Groq-hosted LLM
- Retrieve chunks with BM25, dense embeddings, Reciprocal Rank Fusion and optional Cross-Encoder reranking
- Show the source chunks used for every answer
- Save conversation history and uploaded-document metadata in Supabase/PostgreSQL
- View recent questions, documents, and the active knowledge source in the UI
- Use FAISS for vector search, with a NumPy fallback for Windows development
- Inspect source filenames, headings and retrieval scores with each answer

## How it works

```text
TXT document -> heading-aware token chunks -> BM25 + dense FAISS -> RRF -> optional Cross-Encoder -> Groq -> answer + sources
```

1. A `.txt` document is uploaded through the Next.js interface.
2. The backend splits it into overlapping token chunks, retains headings, and caches normalized Sentence Transformer embeddings by content.
3. BM25 and FAISS (or NumPy) retrieve candidates, RRF combines their ranks, and a Cross-Encoder can rerank them.
4. Up to the configured context budget is sent to Groq with the user's question.
5. The answer and its sources are returned to the UI and saved to PostgreSQL.

> Each new upload replaces the currently active knowledge document, keeping answers tied to the latest file.

Retrieval settings are environment variables: `RAG_CHUNK_TOKENS` (512), `RAG_CHUNK_OVERLAP` (51), `RAG_CHUNK_STRATEGY` (`fixed` or `structure`; default `fixed`), `RAG_CANDIDATE_K` (20), `RAG_RRF_K` (60), `RAG_RERANK` (`false`), `RAG_RETRIEVAL_MODE` (`sparse`, `hybrid`, `dense`, or `baseline`; default `sparse`), `RAG_TOP_K` (5), `RAG_CONTEXT_TOKENS` (2600), `EMBEDDING_MODEL`, and `RERANKER_MODEL`. Set `ALLOW_MODEL_DOWNLOAD=false` to use cached models only. If dense embeddings are unavailable, search falls back to BM25; if the reranker is unavailable, fusion order is retained. Cached embedding files live in ignored `faq-bot/.rag_cache/` and are keyed by source text and configuration. The selected exact FAISS index is rebuilt from those cached vectors at startup.

For hybrid search, `RAG_SPARSE_WEIGHT` (default `1`) controls BM25's RRF contribution. Optional reranking scores only the first `RAG_RERANK_TOP_N` candidates (default `8`) with `RERANKER_MAX_LENGTH` (default `256`); remaining candidates retain fusion order. Increase the length when evidence regularly lies near the end of long chunks.

The `/ask` response includes `source`, `heading`, `score`, `dense_score`, `bm25_score`, and `rerank_score` for each chunk. Send `"debug": true` in the request for a small `retrieval_debug` object. Questions containing a record ID such as `ORC123` return the fallback without a Groq call when that ID is absent from the supplied context. Other insufficient-evidence cases rely on Groq's grounded prompt. The prompt asks for source-number citations and abstention. Scores are retrieval signals, not calibrated probabilities.

## Evaluation

### Follow-up with human-written questions

Run `python evaluation/realistic_benchmark.py --dev-only` to inspect development results, then run `python evaluation/realistic_benchmark.py` once for the held-out comparison. The [benchmark](evaluation/realistic_benchmark.py), [evidence-labeled documents and questions](evaluation/realistic_corpus.py), and [raw results](evaluation/realistic_results.json) are included. The 30 development questions come from three existing hand-labeled company policy documents; eight separate library, transport, housing, museum, and environmental documents contain 25 held-out answerable questions and four unanswerable questions. Every evidence quote is checked against its source before retrieval. Document names and topic labels do not cross splits; a five-word shingle audit found zero overlap between development and held-out documents, and the cross-split question token Jaccard maximum is below 0.85. Existing embedding caches are reused across method comparisons.

| Held-out method | Recall@5 / @10 | MRR@10 | nDCG@10 | p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Previous production: BM25, 512 tokens | 1.00 / 1.00 | **0.960** | **0.970** | **0.04 / 0.08** |
| BM25, 128 tokens | 1.00 / 1.00 | 0.903 | 0.921 | 0.1 / 0.1 |
| Dense, 128 tokens | 0.96 / 1.00 | 0.906 | 0.919 | 6.9 / 7.8 |
| Equal-weight RRF, 128 tokens | 1.00 / 1.00 | 0.888 | 0.910 | 6.6 / 7.1 |
| Hybrid + full Cross-Encoder, 128 tokens | 1.00 / 1.00 | 0.930 | 0.941 | 419 / 441 |
| Hybrid + eight-pair, 256-token Cross-Encoder | 1.00 / 1.00 | 0.930 | 0.941 | **162 / 169** |

The older synthetic hybrid gap came from shared-template distractors: their dense ranks and BM25 ranks combined under equal-weight RRF. A direct rank audit found that all 60 labeled answerable questions had complete evidence in BM25's top five; fusion pushed 29 below rank five, although all 60 remained within the fused top 20. For all 29 displaced questions, dense retrieval lacked complete evidence in its top 20. This is a data/rank-fusion interaction, not candidate-generation loss. A separate filtering bug truncated the fused pool before discarding weak candidates; a focused regression test reproduced it and the repaired index now refills those slots. [Before](evaluation/realistic_results_before_filter.json) and [after](evaluation/realistic_results.json) held-out retrieval quality was unchanged in this small set. On development questions without record IDs, equal-weight RRF improved MRR@10 from BM25's 0.908 to 0.978, but the preselected hybrid method did not beat the existing BM25 configuration on held-out data. The production default therefore remains BM25 with 512-token chunks. The optional fast reranker cut median CPU latency by 61% relative to full reranking on the same held-out queries with identical retrieval metrics; its eight-pair limit and 256-token input are configurable. These documents yield only 22 test chunks, so the saturated Recall@5 values have wide practical limits. The earlier 1.82-second reranker result below used a much larger synthetic index and should not be compared directly with this corpus.

For a small generation check, run `python evaluation/validate_grounding.py` with a Groq key. In the recorded six real [Groq answers](evaluation/grounding_results.json), all six met their case-specific answer checks and used valid source numbers; all five answerable replies cited a labeled supporting chunk, and the unanswerable reply said the document did not specify the requested deposit maximum. This is a functional sample, not a statistical correctness rate. The two initially overstrict wording checks were corrected against saved replies without additional API calls.

### Historical large synthetic stress test

Run `python evaluation/benchmark_advanced.py --smoke`, then `python evaluation/benchmark_advanced.py`. See [methodology](evaluation/advanced_methodology.md) and [raw measured results](evaluation/advanced_results.json). The benchmark uses seed `20261009`: 160 generated long-form documents across eight domains, 128 questions, and disjoint development/test domains. The selected 512-token fixed setup yielded 1,360 chunks for 80 test documents. It compares the legacy dense pipeline with BM25, dense, RRF hybrid and reranked retrieval. The same MiniLM embeddings and test questions are used for the new methods. The benchmark makes no Groq calls.

The table below is a historical measurement from before the explicit-record gate and eight-pair reranker default were added. A fresh run of that script uses the current implementation and may differ; use the realistic benchmark above for the current method comparison.

| Development chunking (36 answerable questions) | Chunks | Dense Recall@5 / evidence coverage@5 | Dense MRR@10 | Warm index ms |
| --- | ---: | ---: | ---: | ---: |
| 256 tokens, 10% overlap, fixed | 408 | 0.583 | 0.252 | 45 |
| 256 tokens, 20% overlap, fixed | 456 | 0.500 | 0.247 | 49 |
| **512 tokens, 10% overlap, fixed (selected)** | **204** | **0.556** | **0.335** | **43** |
| 512 tokens, 20% overlap, fixed | 228 | 0.389 | 0.242 | 46 |
| 1024 tokens, 10% overlap, fixed | 108 | 0.167 | 0.093 | 42 |
| 1024 tokens, 20% overlap, fixed | 120 | 0.556 | 0.275 | 45 |
| 512 tokens, 10% overlap, heading sections | 240 | 0.167 | 0.119 | 41 |
| 512 tokens, 20% overlap, heading sections | 240 | 0.361 | 0.167 | 43 |

Chunking was chosen by development dense MRR@10, then Recall@5. The selected setting gives the best reciprocal rank in that subset with 204 chunks. The fixed strategy still attaches the nearest heading to each chunk; the section strategy also prevents windows from crossing heading boundaries. These warm index times use cached embeddings and show chunk processing/FAISS build cost, not first-time encoding cost.

| Held-out method (60 answerable questions) | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Precision@5 | p50 / p95 retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy dense, 1,000-character chunks | 0.000 | 0.017 | 0.133 | 0.024 | 0.065 | 0.017 | 7.3 / 7.9 |
| BM25 sparse, selected default | 0.333 | 1.000 | 1.000 | 0.583 | 0.796 | 0.333 | 2.3 / 2.7 |
| Dense only | 0.050 | 0.133 | 0.250 | 0.095 | 0.108 | 0.037 | 7.1 / 7.8 |
| RRF hybrid | 0.300 | 0.517 | 0.817 | 0.403 | 0.534 | 0.200 | 9.9 / 11.1 |
| RRF + Cross-Encoder | 0.600 | 0.850 | 0.967 | 0.694 | 0.759 | 0.247 | 1820 / 2151 |

Development selection used evidence coverage@5, then MRR@10 and p95 latency to break ties. BM25 reached 1.000 coverage@5 on the development subset and was selected before test evaluation. The generated questions contain distinctive record identifiers, which strongly favors exact lexical matching; these numbers do not establish performance on arbitrary uploads. Cross-Encoder reranking improved rank-one retrieval but reduced coverage@5 and added substantial CPU latency. The application clips retrieved passages at 2,600 regex tokens; prompt and question tokens are additional.

| FAISS index on 1,360 normalized test vectors | Neighbor recall@10 vs FlatIP | p50 / p95 search ms | Build ms | Serialized bytes |
| --- | ---: | ---: | ---: | ---: |
| FlatIP, selected from development | 1.000 | 0.055 / 0.059 | 0.3 | 2,089,005 |
| HNSW, M=32, efSearch=32 | 0.984 | 0.016 / 0.021 | 41 | 2,458,146 |
| IVF, nlist=34, nprobe=4 | 0.989 | 0.012 / 0.016 | 9 | 2,152,475 |

Development ANN tuning selected FlatIP because its 204-vector index had no material latency disadvantage; the faster ANN results above are held-out observations, not retuning. ANN neighbor recall is separate from evidence recall. The result JSON includes the full `efSearch`, `nprobe`, and `nlist` grid. Timings were measured on Linux with a 12-logical-CPU host, FAISS one thread, Python 3.11.16, NumPy 2.4.6, FAISS 1.15.1, Sentence Transformers 5.7.0 and CPU PyTorch 2.14.1. The first held-out selected-index build took 53.4 seconds including embedding generation; warm development builds were tens of milliseconds, so those indexing times should not be compared as if both were cold. Serialized bytes are an index-size proxy, not process RSS. A separate audit found the maximum cross-split document 5-gram Jaccard similarity was 0.123; the generator also rejects cross-split question token Jaccard at or above 0.85. Answer correctness and unanswerable abstention were not measured with Groq; the four unanswerable test queries all retrieved some text, so the prompt's abstention behavior remains unverified for questions without explicit record IDs.

## Tech stack

| Layer | Tools |
| --- | --- |
| Frontend | Next.js, React, Tailwind CSS |
| Backend | FastAPI, Uvicorn |
| RAG | Sentence Transformers, FAISS, NumPy |
| LLM | Groq API (configurable model) |
| Database | Supabase PostgreSQL |

## Project structure

```text
.
|- faq-bot/          # FastAPI backend, RAG pipeline, database schema
|- frontend/         # Next.js chat interface
|- backend_faq-bot.md
`- README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/keys)
- A Supabase PostgreSQL project and connection string

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/Uv0503/DocuMind-RAG-Based-FAQ-Assistant.git
cd DocuMind-RAG-Based-FAQ-Assistant
```

### 2. Configure the backend

Create `faq-bot/.env`:

```env
DATABASE_URL=your_supabase_postgres_connection_string
GROQ_API_KEY=your_groq_api_key
# Optional; defaults to openai/gpt-oss-20b
GROQ_MODEL=openai/gpt-oss-20b
```

Install and start the API:

```bash
cd faq-bot
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8888
```

The API will be available at `http://localhost:8888`, and interactive API documentation is available at `http://localhost:8888/docs`.

### 3. Configure the database

In the Supabase SQL Editor, run the contents of [`faq-bot/schema.sql`](faq-bot/schema.sql).

### 4. Configure and start the frontend

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8888
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`, upload a `.txt` file, and start asking questions.

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API welcome message |
| `GET` | `/health/db` | Database health check |
| `GET` | `/knowledge` | Active source and indexed chunk count |
| `POST` | `/upload` | Upload and index a `.txt` document |
| `POST` | `/ask` | Ask a question about the active document |
| `GET` | `/conversations` | Saved question-and-answer history |
| `GET` | `/documents` | Uploaded document history |

Example question request:

```bash
curl -X POST http://localhost:8888/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does this document say about pricing?\"}"
```

## Environment and security

- Keep `GROQ_API_KEY` and `DATABASE_URL` in local `.env` files only.
- Do not commit `.env`, `.env.local`, virtual environments, `node_modules`, or uploaded documents.
- The repository `.gitignore` already excludes these local and sensitive files.

## Remaining limitations

Uploads still accept one active UTF-8 TXT document. Heading metadata has no page numbers. The default abstention decision depends on retrieved evidence and Groq's grounded prompt; it is not a calibrated factuality guarantee.

## Created by

**Aman Singh Chauhan**
[GitHub: @Uv0503](https://github.com/Uv0503)
