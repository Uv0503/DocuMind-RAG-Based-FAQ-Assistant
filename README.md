# DocuMind - RAG-Based FAQ Assistant

**DocuMind** is a full-stack, document-grounded FAQ assistant. Upload a UTF-8 `.txt` document, ask a question, and receive an answer generated from the most relevant parts of that document -- along with the source chunks used to create it.

[View the project on GitHub](https://github.com/Uv0503/DocuMind-RAG-Based-FAQ-Assistant)

## Features

- Upload and index a `.txt` knowledge document
- Generate grounded answers with Groq LLaMA
- Retrieve relevant document chunks with semantic search
- Show the source chunks used for every answer
- Save conversation history and uploaded-document metadata in Supabase/PostgreSQL
- View recent questions, documents, and the active knowledge source in the UI
- Use FAISS for vector search, with a NumPy fallback for Windows development

## How it works

```text
TXT document -> chunking -> embeddings -> vector search -> relevant chunks -> Groq LLaMA -> answer + sources
```

1. A `.txt` document is uploaded through the Next.js interface.
2. The backend splits it into overlapping chunks and generates embeddings with `sentence-transformers`.
3. FAISS (or the Windows NumPy fallback) identifies the chunks most relevant to the question.
4. Those chunks are sent to Groq LLaMA with the user's question.
5. The answer and its sources are returned to the UI and saved to PostgreSQL.

> Each new upload replaces the currently active knowledge document, keeping answers tied to the latest file.

## Tech stack

| Layer | Tools |
| --- | --- |
| Frontend | Next.js, React, Tailwind CSS |
| Backend | FastAPI, Uvicorn |
| RAG | Sentence Transformers, FAISS, NumPy |
| LLM | Groq LLaMA |
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

## Future improvements

- Support PDF, DOCX, and multiple-document uploads
- Add user authentication and per-user document libraries
- Store embeddings in a persistent vector database
- Add citations that include document names and page/section information

## Created by

**Aman Singh Chauhan**
[GitHub: @Uv0503](https://github.com/Uv0503)
