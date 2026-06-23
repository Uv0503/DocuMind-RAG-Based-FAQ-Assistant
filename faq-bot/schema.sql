-- Stores every user question, model answer, retrieved context, and source chunks.
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

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
