"""FastAPI RAG prototype with guardrails and configurable similarity metrics."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .corpus import CorpusIndex, Document, load_corpus
from .guardrails import check_guardrails


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User query to answer")
    k: int = Field(3, ge=1, le=10, description="Number of snippets to return")
    similarity: Literal["cosine", "dot"] = Field("cosine", description="Similarity metric to use")


class Snippet(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    snippets: list[Snippet]
    guardrail_triggered: bool
    guardrail_reason: str | None = None
    index_config: dict


app = FastAPI(title="Palm RAG Prototype", version="0.1.0")

_corpus = load_corpus()
_index = CorpusIndex(_corpus)


def compose_answer(docs: list[tuple[Document, float]]) -> str:
    if not docs:
        return "No supporting documents found."
    # Simple summarisation: take the highest score snippet and append provenance sentence.
    top_doc, _ = docs[0]
    supporting_titles = ", ".join(doc.title for doc, _ in docs[:3])
    return (
        f"{top_doc.text} (Answer grounded in: {supporting_titles})."
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest) -> AnswerResponse:
    guardrail = check_guardrails(request.query)
    if guardrail.triggered:
        return AnswerResponse(
            answer="Unable to answer: query violates safety guardrails.",
            snippets=[],
            guardrail_triggered=True,
            guardrail_reason=guardrail.reason,
            index_config={"similarity": request.similarity, "k": 0},
        )

    docs = _index.search(request.query, top_k=request.k, similarity=request.similarity)
    snippets = [
        Snippet(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=score)
        for doc, score in docs
    ]
    answer_text = compose_answer(docs)

    return AnswerResponse(
        answer=answer_text,
        snippets=snippets,
        guardrail_triggered=False,
        guardrail_reason=None,
        index_config={"similarity": request.similarity, "k": request.k},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "documents": str(len(_corpus))}