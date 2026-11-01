# API composition root: this module wires HTTP routes to the RAG, model, and database layers.
# Request flow: Next.js client -> FastAPI validation -> retrieval/model work -> JSON response.
import asyncio
import os
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from db import (create_session, create_user, db, ensure_schema, fetch_conversations,
                fetch_documents, find_session_user, find_user_by_email, log_document,
                log_interaction, require_db, verify_password)
from model import answer_question
from rag import bound_sources, evidence_sufficient, rag_index, save_uploaded_text


bearer_scheme = HTTPBearer(auto_error=False)


# Single fallback used when no knowledge has been indexed yet.
FALLBACK_ANSWER = "OOPS!!😥 I could not find enough information in the uploaded knowledge base to answer this confidently."

# FastAPI metadata appears in Swagger at /docs.
app = FastAPI(
    title="DocuMind - RAG-Based FAQ Assistant",
    description="DocuMind, created by Aman Singh Chauhan. A RAG-based FAQ assistant powered by FastAPI, FAISS, and the Groq API",
    contact={
        "name": "Aman Singh Chauhan",
        "email": "picture0503@gmail.com",
    },
)

# Allow local development plus deployment origins configured as a comma-separated list.
configured_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_URL", "https://llm-faq-bot.vercel.app").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", *configured_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    # Keep the public ask payload simple: the user only sends a question.
    question: str = Field(..., min_length=3, max_length=500)
    debug: bool = False

    @field_validator("question", mode="before")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        # Strip whitespace early so validation and model calls receive the clean question.
        if not isinstance(value, str):
            return value
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()

class Credentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        email = value.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise ValueError("Enter a valid email address")
        return email


class AuthResponse(BaseModel):
    token: str
    user: dict


async def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Sign in is required")
    try:
        user = await find_session_user(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication database is unavailable") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again")
    return dict(user)


# Response schemas document the public API contract and are also exposed in /docs.
class Source(BaseModel):
    # Source chunks are returned with every grounded answer so users can inspect the evidence.
    chunk_id: int
    source: str
    heading: str = ""
    text: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Answer(BaseModel):
    answer: str
    sources: List[Source]
    retrieval_debug: Optional[dict] = None


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
        try:
            # Managed Postgres services can take several seconds to wake up and run
            # the initial schema checks. Keep the API startup resilient to that latency.
            db_timeout = float(os.getenv("DATABASE_STARTUP_TIMEOUT", "30"))
            await asyncio.wait_for(db.connect(), timeout=db_timeout)
            await ensure_schema()
            print("Connected to the database successfully.")
        except Exception as exc:
            print(f"Database unavailable. Database features are disabled: {exc}")
    else:
        print("DATABASE_URL is not configured. Database features are disabled.")


@app.on_event("shutdown")
async def shutdown():
    # Close the database connection cleanly when the API server stops.
    if db is not None and db.is_connected:
        await db.disconnect()


@app.get("/")
def root():
    # Tiny landing endpoint so visiting the API root does not return a 404.
    return {"message": "Welcome to DocuMind, created by Aman Singh Chauhan. Visit /docs to test it."}


@app.get("/health/db", response_model=List[str])
async def health_db():
    # Quick database check used during local setup and demos.
    try:
        database = await require_db()
        rows = await database.fetch_all(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not connected") from exc
    return [r["table_name"] for r in rows]


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
async def register(credentials: Credentials):
    try:
        user = await create_user(credentials.email, credentials.password)
        token = await create_session(user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication database is unavailable") from exc
    return {"token": token, "user": {"id": user["id"], "email": user["email"]}}


@app.post("/auth/login", response_model=AuthResponse)
async def login(credentials: Credentials):
    try:
        user = await find_user_by_email(credentials.email)
        if user is None or not verify_password(credentials.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        token = await create_session(user["id"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Authentication database is unavailable") from exc
    return {"token": token, "user": {"id": user["id"], "email": user["email"]}}


@app.get("/auth/me")
async def get_me(user: dict = Depends(current_user)):
    return {"user": user}


@app.get("/conversations", response_model=List[Conversation])
async def list_conversations(user: dict = Depends(current_user)):
    # Return stored question/answer history from Postgres.
    try:
        rows = await fetch_conversations(user["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return [dict(r) for r in rows]


@app.get("/documents", response_model=List[DocumentRecord])
async def list_documents(user: dict = Depends(current_user)):
    # Return uploaded document history from Postgres.
    try:
        return await fetch_documents(user["id"])
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
async def ask(query: AskRequest, user: dict = Depends(current_user)):
    # This endpoint is the RAG orchestration boundary: retrieve first, generate second,
    # then persist the evidence and result for later inspection.
    # Retrieve the most relevant chunks before calling the LLM.
    sources = rag_index.retrieve(query.question, top_k=int(os.getenv("RAG_TOP_K", "5")))
    sources = bound_sources(sources, int(os.getenv("RAG_CONTEXT_TOKENS", "2600")))

    # If no documents exist or no chunks pass the relevance threshold, return and log the fallback.
    if not evidence_sufficient(query.question, sources):
        try:
            await log_interaction(user["id"], query.question, "", FALLBACK_ANSWER, [])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")
        return {"answer": FALLBACK_ANSWER, "sources": [],
                "retrieval_debug": {"candidates": len(sources),
                                    "reason": "missing_evidence_or_record_identifier"} if query.debug else None}

    public_sources = [
        {key: source.get(key) for key in ("chunk_id", "source", "heading", "text",
                                         "score", "dense_score", "bm25_score", "rerank_score")}
        for source in sources
    ]

    # Store the exact retrieved text in the conversation history for auditability.
    context = "\n\n".join(source["text"] for source in sources)

    try:
        # The configured Groq model generates the final answer using only the retrieved source chunks.
        answer = answer_question(query.question, sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Model inference failed")

    if not answer:
        raise HTTPException(status_code=500, detail="Model failed to generate an answer")

    try:
        # Log both the answer and its source context in Postgres.
        await log_interaction(user["id"], query.question, context, answer, public_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")

    return {"answer": answer, "sources": public_sources,
            "retrieval_debug": {"mode": os.getenv("RAG_RETRIEVAL_MODE", "sparse"),
                                "candidates": len(sources), "dense_available": rag_index.embeddings is not None}
            if query.debug else None}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user: dict = Depends(current_user)):
    # Uploads arrive as multipart/form-data rather than JSON because they contain file bytes.
    # Keep upload scope narrow: .txt only.
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()
    try:
        # Save the latest upload, rebuild the vector index, and record the upload in DB.
        text = content.decode("utf-8").strip()
        chunks_created = save_uploaded_text(file.filename, content)
        rag_index.rebuild()
        await log_document(user["id"], file.filename, text, chunks_created)
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
