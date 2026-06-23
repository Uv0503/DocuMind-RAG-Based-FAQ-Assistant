import os
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from db import db, ensure_schema, fetch_conversations, fetch_documents, log_document, log_interaction
from model import answer_question
from rag import rag_index, save_uploaded_text


# Single fallback used when no knowledge has been indexed yet.
FALLBACK_ANSWER = "OOPS!!😥 I could not find enough information in the uploaded knowledge base to answer this confidently."

# FastAPI metadata appears in Swagger at /docs.
app = FastAPI(
    title="DocuMind - RAG-Based FAQ Assistant",
    description="DocuMind, created by Aman Singh Chauhan. A RAG-based FAQ assistant powered by FastAPI, FAISS, and Groq LLaMA",
    contact={
        "name": "Aman Singh Chauhan",
        "email": "picture0503@gmail.com",
    },
)

# Allow the local Next.js frontend and deployed frontend to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://llm-faq-bot.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    # Keep the public ask payload simple: the user only sends a question.
    question: str = Field(..., min_length=3, max_length=500)

    @field_validator("question", mode="before")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        # Strip whitespace early so validation and model calls receive the clean question.
        if not isinstance(value, str):
            return value
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()

class Source(BaseModel):
    # Source chunks are returned with every grounded answer so users can inspect the evidence.
    chunk_id: int
    text: str


class Answer(BaseModel):
    answer: str
    sources: List[Source]


class Conversation(BaseModel):
    # Shape returned by GET /conversations.
    id: int
    question: str
    context: Optional[str] = None
    response: str
    sources: Optional[List[dict]] = None
    created_at: datetime


class DocumentRecord(BaseModel):
    # Upload history response. The full content is stored in DB but not returned here.
    id: int
    filename: str
    chunks_created: int
    created_at: datetime


class KnowledgeStatus(BaseModel):
    # Small debug response showing the currently indexed knowledge file.
    chunks_indexed: int
    active_sources: List[str]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Convert FastAPI's default 422 validation errors to simpler 400 errors for this beginner project.
    return JSONResponse(status_code=400, content=jsonable_encoder({"detail": exc.errors()}))


@app.on_event("startup")
async def startup():
    # Build the vector index once at startup so the first question is ready to retrieve context.
    rag_index.rebuild()
    if db is not None:
        await db.connect()
        await ensure_schema()
        print("Connected to the database successfully.")
    else:
        print("DATABASE_URL is not configured. Database features are disabled.")


@app.on_event("shutdown")
async def shutdown():
    # Close the database connection cleanly when the API server stops.
    if db is not None:
        await db.disconnect()


@app.get("/")
def root():
    # Tiny landing endpoint so visiting the API root does not return a 404.
    return {"message": "Welcome to DocuMind, created by Aman Singh Chauhan. Visit /docs to test it."}


@app.get("/health/db", response_model=List[str])
async def health_db():
    # Quick database check used during local setup and demos.
    if db is None:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")

    rows = await db.fetch_all(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    return [r["table_name"] for r in rows]


@app.get("/conversations", response_model=List[Conversation])
async def list_conversations():
    # Return stored question/answer history from Postgres.
    try:
        rows = await fetch_conversations()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return [dict(r) for r in rows]


@app.get("/documents", response_model=List[DocumentRecord])
async def list_documents():
    # Return uploaded document history from Postgres.
    try:
        return await fetch_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/knowledge", response_model=KnowledgeStatus)
async def knowledge_status():
    # Show which source file is currently powering RAG retrieval.
    return {
        "chunks_indexed": rag_index.chunk_count(),
        "active_sources": rag_index.active_sources(),
    }


@app.post("/ask", response_model=Answer)
async def ask(query: AskRequest):
    # Retrieve the most relevant chunks before calling the LLM.
    sources = rag_index.retrieve(query.question, top_k=3)

    # If no documents exist or no chunks pass the relevance threshold, return and log the fallback.
    if not sources:
        try:
            await log_interaction(query.question, "", FALLBACK_ANSWER, [])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")
        return {"answer": FALLBACK_ANSWER, "sources": []}

    # Remove internal fields like score/source filename from the public API response.
    public_sources = [
        {"chunk_id": source["chunk_id"], "text": source["text"]}
        for source in sources
    ]

    # Store the exact retrieved text in the conversation history for auditability.
    context = "\n\n".join(source["text"] for source in sources)

    try:
        # Groq generates the final answer using only the retrieved source chunks.
        answer = answer_question(query.question, sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Model inference failed")

    if not answer:
        raise HTTPException(status_code=500, detail="Model failed to generate an answer")

    try:
        # Log both the answer and its source context in Postgres.
        await log_interaction(query.question, context, answer, public_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")

    return {"answer": answer, "sources": public_sources}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Keep upload scope narrow: .txt only.
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()
    try:
        # Save the latest upload, rebuild the vector index, and record the upload in DB.
        text = content.decode("utf-8").strip()
        chunks_created = save_uploaded_text(file.filename, content)
        rag_index.rebuild()
        await log_document(file.filename, text, chunks_created)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file must be valid UTF-8 text")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save and index uploaded document")

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "chunks_created": chunks_created,
        "active_sources": rag_index.active_sources(),
    }


if __name__ == "__main__":
    # Local development entrypoint; production can still run uvicorn directly.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8888)))
