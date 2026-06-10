"""
PaperPilot FastAPI Backend
Serves the agent with SSE streaming support.

Usage:
    pip install fastapi uvicorn sse-starlette
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Place this file at: C:/Users/dhrup/OneDrive/Desktop/PaperPilot/api.py
"""

import os
import sys
import json
import asyncio
import sqlite3
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

# Add agent src paths
sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "data" / "paperpilot.db"

app = FastAPI(
    title="PaperPilot API",
    description="Multi-hop RAG over NLP/ML research papers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    max_iterations: int = 2


class PaperMeta(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    citation_count: int
    abstract: str
    s2_tldr: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_paper_meta(arxiv_id: str) -> dict:
    """Fetch paper metadata from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT arxiv_id, title, authors, year,
                      citation_count, abstract, s2_tldr
               FROM papers WHERE arxiv_id = ?""",
            (arxiv_id,)
        ).fetchone()
        if not row:
            return {}
        d = dict(row)
        try:
            d["authors"] = json.loads(d.get("authors") or "[]")
        except Exception:
            d["authors"] = []
        return d
    finally:
        conn.close()


def get_corpus_stats() -> dict:
    """Return corpus statistics."""
    conn = sqlite3.connect(DB_PATH)
    try:
        papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
        return {"papers": papers, "chunks": chunks}
    finally:
        conn.close()


async def run_agent_streaming(query: str, max_iterations: int) -> AsyncGenerator[str, None]:
    """
    Run the PaperPilot agent and stream SSE events.

    Event types:
        status   — pipeline step updates (decomposing, retrieving, etc.)
        token    — answer text token (streamed word by word)
        citation — a cited paper's metadata
        done     — final result summary
        error    — error message
    """
    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        yield sse("status", {"message": "Analyzing your question...", "step": "start"})
        await asyncio.sleep(0.05)

        # Import here to avoid startup cost if not needed
        from graph import run_agent

        yield sse("status", {"message": "Decomposing into sub-questions...", "step": "decompose"})
        await asyncio.sleep(0.05)

        # Run agent in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_agent(query, max_iterations=max_iterations)
        )

        # Stream answer word by word
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        verification_status = result.get("verification_status", "unknown")
        sub_questions = result.get("sub_questions", [])
        iterations_used = result.get("iterations_used", 0)
        question_type = result.get("question_type", "")

        yield sse("status", {"message": "Streaming answer...", "step": "stream"})
        await asyncio.sleep(0.05)

        # Stream token by token (word-level)
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield sse("token", {"text": token})
            await asyncio.sleep(0.015)  # ~67 words/sec

        # Send citation cards
        yield sse("status", {"message": "Loading paper details...", "step": "citations"})
        for arxiv_id in citations:
            meta = get_paper_meta(arxiv_id)
            if meta:
                yield sse("citation", {
                    "arxiv_id": arxiv_id,
                    "title": meta.get("title", arxiv_id),
                    "authors": meta.get("authors", [])[:3],
                    "year": meta.get("year", 0),
                    "citation_count": meta.get("citation_count", 0),
                    "abstract": meta.get("abstract", "")[:300],
                    "s2_tldr": meta.get("s2_tldr", ""),
                    "url": f"https://arxiv.org/abs/{arxiv_id}"
                })
                await asyncio.sleep(0.02)

        # Final summary
        yield sse("done", {
            "verification_status": verification_status,
            "citations_count": len(citations),
            "iterations_used": iterations_used,
            "question_type": question_type,
            "sub_questions": sub_questions,
        })

    except Exception as e:
        yield sse("error", {"message": str(e)})


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "PaperPilot API", "status": "ok", "version": "1.0.0"}


@app.get("/stats")
def stats():
    """Return corpus statistics."""
    return get_corpus_stats()


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Stream a query response via SSE.
    Events: status | token | citation | done | error
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        run_agent_streaming(req.query.strip(), req.max_iterations),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/papers/{arxiv_id}")
def get_paper(arxiv_id: str):
    """Fetch metadata for a single paper."""
    meta = get_paper_meta(arxiv_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    return meta


@app.get("/health")
def health():
    stats = get_corpus_stats()
    return {"status": "healthy", "corpus": stats}