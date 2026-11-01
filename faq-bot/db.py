# Persistence layer: provides optional PostgreSQL history without coupling RAG availability to the DB.
# When no live connection exists, logging/list operations degrade gracefully to no-ops or empty lists.
import asyncio
import os
import json
import hashlib
import secrets
from typing import List

from databases import Database
from dotenv import load_dotenv


load_dotenv()

# Database URL comes from .env so secrets are never placed in frontend code.
DATABASE_URL = os.getenv("DATABASE_URL")
db = Database(DATABASE_URL) if DATABASE_URL else None
_db_connect_lock = asyncio.Lock()


async def require_db() -> Database:
    """Return a live database connection, reconnecting after transient outages.

    Render and Supabase can independently sleep or recycle connections. A failed
    startup connection must therefore not leave authentication disabled until the
    next application restart.
    """
    if db is None:
        raise RuntimeError("DATABASE_URL is not configured")

    if not db.is_connected:
        async with _db_connect_lock:
            if not db.is_connected:
                timeout = float(os.getenv("DATABASE_STARTUP_TIMEOUT", "30"))
                await asyncio.wait_for(db.connect(), timeout=timeout)
    return db


async def ensure_schema():
    # Create or update the small tables needed for the demo when the API starts.
    database = await require_db()
    await database.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
    """)
    await database.execute("""
    CREATE TABLE IF NOT EXISTS user_sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash TEXT NOT NULL UNIQUE,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
    """)
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
    await database.execute("ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
    await database.execute("CREATE INDEX IF NOT EXISTS conversations_user_created_idx ON conversations(user_id, created_at DESC)")
    await database.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        chunks_created INTEGER NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    )
    """)
    await database.execute("ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
    await database.execute("CREATE INDEX IF NOT EXISTS documents_user_created_idx ON documents(user_id, created_at DESC)")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = stored.split("$", 2)
        if algorithm != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=16384, r=8, p=1)
        return secrets.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_user(email: str, password: str):
    database = await require_db()
    existing = await database.fetch_one("SELECT id FROM users WHERE email = :email", {"email": email})
    if existing:
        raise ValueError("An account with this email already exists")
    return await database.fetch_one(
        "INSERT INTO users(email, password_hash) VALUES(:email, :password_hash) RETURNING id, email, created_at",
        {"email": email, "password_hash": hash_password(password)},
    )


async def find_user_by_email(email: str):
    return await (await require_db()).fetch_one("SELECT id, email, password_hash FROM users WHERE email = :email", {"email": email})


async def create_session(user_id: int):
    token = secrets.token_urlsafe(32)
    await (await require_db()).execute(
        "INSERT INTO user_sessions(user_id, token_hash, expires_at) VALUES(:user_id, :token_hash, NOW() + INTERVAL '7 days')",
        {"user_id": user_id, "token_hash": hash_token(token)},
    )
    return token


async def find_session_user(token: str):
    return await (await require_db()).fetch_one(
        """SELECT users.id, users.email FROM user_sessions
           JOIN users ON users.id = user_sessions.user_id
           WHERE user_sessions.token_hash = :token_hash AND user_sessions.expires_at > NOW()""",
        {"token_hash": hash_token(token)},
    )


async def log_interaction(
    user_id: int,
    question: str,
    context: str,
    response: str,
    sources: List[dict],
):
    # Store the question, final answer, retrieved context, and source chunks for interview/demo review.
    if db is None:
        return
    query = """
    INSERT INTO conversations(user_id, question, context, response, sources)
    VALUES(:user_id, :question, :context, :response, CAST(:sources AS JSONB))
    """
    await (await require_db()).execute(query=query, values={
        "user_id": user_id,
        "question": question,
        "context": context,
        "response": response,
        "sources": json.dumps(sources),
    })


async def log_document(user_id: int, filename: str, content: str, chunks_created: int):
    # Keep an upload history in Postgres while the active RAG file stays on disk.
    if db is None:
        return
    query = """
    INSERT INTO documents(user_id, filename, content, chunks_created)
    VALUES(:user_id, :filename, :content, :chunks_created)
    """
    await (await require_db()).execute(query=query, values={
        "user_id": user_id,
        "filename": filename,
        "content": content,
        "chunks_created": chunks_created,
    })


async def fetch_documents(user_id: int):
    # Return metadata only; full document text stays in the table for inspection if needed.
    if db is None:
        return []
    rows = await (await require_db()).fetch_all(
        "SELECT id, filename, chunks_created, created_at FROM documents WHERE user_id = :user_id ORDER BY created_at DESC",
        {"user_id": user_id},
    )
    return [dict(row) for row in rows]


async def fetch_conversations(user_id: int):
    # Newest interactions first makes the history easier to inspect in the UI or Swagger.
    if db is None:
        return []
    rows = await (await require_db()).fetch_all("SELECT * FROM conversations WHERE user_id = :user_id ORDER BY created_at DESC", {"user_id": user_id})

    conversations = []
    for row in rows:
        conversation = dict(row)
        # Some drivers return JSONB as a string, so normalize it before FastAPI serializes the response.
        if isinstance(conversation.get("sources"), str):
            conversation["sources"] = json.loads(conversation["sources"])
        conversations.append(conversation)
    return conversations
