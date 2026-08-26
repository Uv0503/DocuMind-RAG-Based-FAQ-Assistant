-- Persistence schema for the audit/history side of DocuMind.
-- The active vector index is not stored here; it is rebuilt in memory from the latest text upload.

-- Accounts use scrypt password hashes; plaintext passwords are never stored.
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Opaque session tokens are hashed before storage and expire after seven days.
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Stores every user question, model answer, retrieved context, and source chunks.
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS conversations_user_created_idx ON conversations(user_id, created_at DESC);

-- Keeps old deployments compatible if context used to be required.
ALTER TABLE IF EXISTS conversations
    ALTER COLUMN context DROP NOT NULL;

-- Stores the source chunks shown with each answer.
ALTER TABLE IF EXISTS conversations
    ADD COLUMN IF NOT EXISTS sources JSONB;

-- Stores upload history so uploaded documents can be inspected from the database.
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    chunks_created INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE IF EXISTS documents
    ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS documents_user_created_idx ON documents(user_id, created_at DESC);
