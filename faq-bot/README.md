# DocuMind FastAPI Backend

This backend powers DocuMind, created by Aman Singh Chauhan.

It loads the latest uploaded `.txt` file, chunks the text, embeds chunks with `sentence-transformers`, searches them with FAISS, and sends the top matching chunks to Groq LLaMA for final answer generation. If no upload exists, it uses `context.txt` as the default knowledge base.

## Environment

Create `faq-bot/.env`:

```env
DATABASE_URL=your_supabase_postgres_connection_string
GROQ_API_KEY=your_groq_api_key
```

## Run

```bash
cd faq-bot
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8888
```

## Endpoints

- `GET /` - API welcome message.
- `GET /health/db` - lists public database tables.
- `GET /conversations` - returns chat history.
- `GET /documents` - returns uploaded document records.
- `GET /knowledge` - shows the currently active indexed source.
- `POST /upload` - uploads and indexes a `.txt` knowledge file.
- `POST /ask` - retrieves relevant chunks, asks Groq LLaMA, logs the interaction, and returns answer plus sources.

## Ask Request

```json
{
  "question": "What database is used?"
}
```

## Ask Response

```json
{
  "answer": "The project uses Supabase/PostgreSQL for conversation history.",
  "sources": [
    {
      "chunk_id": 1,
      "text": "..."
    }
  ]
}
```

## Upload Response

```json
{
  "message": "Document uploaded and indexed successfully",
  "filename": "faq.txt",
  "chunks_created": 12,
  "active_sources": ["faq.txt"]
}
```

Each new upload replaces the previously uploaded `.txt` knowledge file. This keeps the demo simple and prevents old uploads from affecting new answers.

## Supabase Migration

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE IF EXISTS conversations
    ALTER COLUMN context DROP NOT NULL;

ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS sources JSONB;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    chunks_created INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
