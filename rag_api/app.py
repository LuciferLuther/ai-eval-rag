"""FastAPI RAG prototype with guardrails and configurable similarity metrics."""
from __future__ import annotations

import time
import numpy as np
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .corpus import CorpusIndex, Document, load_corpus
from .guardrails import check_guardrails

# ---------------------------
# Request/Response Models
# ---------------------------
class AnswerRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    k: int = Field(default=3, ge=1, le=10, description="Number of top documents to retrieve")
    similarity: Literal["cosine", "dot"] = Field(default="cosine", description="Similarity metric to use")

class Snippet(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    snippets: list[Snippet]
    guardrail_triggered: bool
    guardrail_reason: str | None
    index_config: dict

app = FastAPI(title="Palm RAG Prototype", version="0.1.0")

_corpus = load_corpus()
_index = CorpusIndex(_corpus)

# ---------------------------
# Lightweight monitoring
# ---------------------------
STATS = {
    "requests": 0,
    "blocked": 0,
    "p95_latency_ms": 0.0,
    "hit_rate_over_0_2": 0.0,
    "lat_hist_ms": [],  # in-memory histogram
}

def update_stats(lat_ms: float, top_score: float, blocked: bool) -> None:
    STATS["requests"] += 1
    if blocked:
        STATS["blocked"] += 1
    STATS["lat_hist_ms"].append(lat_ms)
    arr = np.array(STATS["lat_hist_ms"], dtype=float)
    # Robust p95 for small N: fall back to max if fewer than 5
    STATS["p95_latency_ms"] = float(np.percentile(arr, 95)) if arr.size >= 5 else float(arr.max(initial=0.0))
    # Crude quality proxy: top-score >= 0.2 -> "hit"
    hits = getattr(update_stats, "_hits", 0)
    total = getattr(update_stats, "_total", 0)
    hits += int(top_score >= 0.2)
    total += 1
    update_stats._hits, update_stats._total = hits, total
    STATS["hit_rate_over_0_2"] = (hits / total) if total else 0.0

# ---------------------------
# Answer composition
# ---------------------------
def compose_answer(docs: list[tuple[Document, float]]) -> str:
    if not docs:
        return "No supporting documents found."
    top_doc, _ = docs[0]
    supporting_titles = ", ".join(doc.title for doc, _ in docs[:3])
    return f"{top_doc.text} (Answer grounded in: {supporting_titles})."

# ---------------------------
# Routes
# ---------------------------
@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest) -> AnswerResponse:
    t0 = time.time()

    guardrail = check_guardrails(request.query)
    if guardrail.triggered:
        lat = round((time.time() - t0) * 1000.0, 2)
        update_stats(lat_ms=lat, top_score=0.0, blocked=True)
        return AnswerResponse(
            answer="Unable to answer: query violates safety guardrails.",
            snippets=[],
            guardrail_triggered=True,
            guardrail_reason=guardrail.reason,
            index_config={"similarity": request.similarity, "k": 0},
        )

    docs = _index.search(request.query, top_k=request.k, similarity=request.similarity)
    snippets = [Snippet(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=score) for doc, score in docs]
    answer_text = compose_answer(docs)

    top_score = float(docs[0][1]) if docs else 0.0
    lat = round((time.time() - t0) * 1000.0, 2)
    update_stats(lat_ms=lat, top_score=top_score, blocked=False)

    return AnswerResponse(
        answer=answer_text,
        snippets=snippets,
        guardrail_triggered=False,
        guardrail_reason=None,
        index_config={"similarity": request.similarity, "k": request.k},
    )

@app.get("/metrics")
async def metrics() -> dict:
    # Don't return the full histogram to keep payload small
    return {
        "requests": STATS["requests"],
        "blocked": STATS["blocked"],
        "p95_latency_ms": STATS["p95_latency_ms"],
        "hit_rate_over_0_2": STATS["hit_rate_over_0_2"],
    }

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "documents": str(len(_corpus))}
