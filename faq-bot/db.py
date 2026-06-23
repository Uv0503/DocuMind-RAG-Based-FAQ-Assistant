import os
import json
from typing import List

from databases import Database
from dotenv import load_dotenv


load_dotenv()

# Database URL comes from .env so secrets are never placed in frontend code.
DATABASE_URL = os.getenv("DATABASE_URL")
db = Database(DATABASE_URL) if DATABASE_URL else None


def require_db() -> Database:
    # Central guard gives every DB caller the same clear error when DATABASE_URL is missing.
    if db is None:
        raise RuntimeError("DATABASE_URL is not configured")
    return db


async def ensure_schema():
    # Create or update the small tables needed for the demo when the API starts.
    database = require_db()
    await database.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        context TEXT,
        response TEXT NOT NULL,
        sources JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
    """)
    await database.execute("ALTER TABLE IF EXISTS conversations ALTER COLUMN context DROP NOT NULL")
    await database.execute("ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS sources JSONB")
    await database.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        chunks_created INTEGER NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
    """)


async def log_interaction(
    question: str,
    context: str,
    response: str,
    sources: List[dict],
):
    # Store the question, final answer, retrieved context, and source chunks for interview/demo review.
    query = """
    INSERT INTO conversations(question, context, response, sources)
    VALUES(:question, :context, :response, CAST(:sources AS JSONB))
    """
    await require_db().execute(query=query, values={
        "question": question,
        "context": context,
        "response": response,
        "sources": json.dumps(sources),
    })


async def log_document(filename: str, content: str, chunks_created: int):
    # Keep an upload history in Postgres while the active RAG file stays on disk.
    query = """
    INSERT INTO documents(filename, content, chunks_created)
    VALUES(:filename, :content, :chunks_created)
    """
    await require_db().execute(query=query, values={
        "filename": filename,
        "content": content,
        "chunks_created": chunks_created,
    })


async def fetch_documents():
    # Return metadata only; full document text stays in the table for inspection if needed.
    rows = await require_db().fetch_all(
        "SELECT id, filename, chunks_created, created_at FROM documents ORDER BY created_at DESC"
    )
    return [dict(row) for row in rows]


async def fetch_conversations():
    # Newest interactions first makes the history easier to inspect in the UI or Swagger.
    rows = await require_db().fetch_all("SELECT * FROM conversations ORDER BY created_at DESC")

    conversations = []
    for row in rows:
        conversation = dict(row)
        # Some drivers return JSONB as a string, so normalize it before FastAPI serializes the response.
        if isinstance(conversation.get("sources"), str):
            conversation["sources"] = json.loads(conversation["sources"])
        conversations.append(conversation)
    return conversations
