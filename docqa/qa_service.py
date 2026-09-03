"""HTTP service: a question about the toolchain in, a grounded answer out.

The decision this file owns is the abstention rule. Retrieval always returns
something, so the service refuses to answer unless the reranked top passage
clears a floor and the shortlist agrees on a document.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from docqa.doc_index import InfraiError, rerank, search

RELEVANCE_FLOOR = 0.35
DEFAULT_COLLECTION = "devtools-docs"


class AskRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)
    collection: str = DEFAULT_COLLECTION
    kind: Literal["build_event", "release_op", "diagnostic"] | None = None
    max_passages: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    doc_id: str
    kind: str
    relevance: float
    excerpt: str


class AskResponse(BaseModel):
    verdict: Literal["answered", "insufficient_evidence"]
    primary_doc: str | None
    citations: list[Citation]


def decide(ranked: Sequence[dict[str, Any]], *, floor: float = RELEVANCE_FLOOR) -> AskResponse:
    """Answer only when the best passage clears the floor; otherwise abstain."""
    citations = [
        Citation(
            doc_id=match["metadata"]["doc_id"],
            kind=match["metadata"]["kind"],
            relevance=round(float(match["relevance"]), 4),
            excerpt=match["metadata"]["text"][:280],
        )
        for match in ranked
    ]
    if not citations or citations[0].relevance < floor:
        return AskResponse(verdict="insufficient_evidence", primary_doc=None, citations=citations)
    return AskResponse(verdict="answered", primary_doc=citations[0].doc_id, citations=citations)


app = FastAPI(title="devtools doc QA")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> Any:
    try:
        matches = search(request.collection, request.question, kind=request.kind)
        ranked = rerank(request.question, matches, top_k=request.max_passages)
    except InfraiError as exc:
        # A rejected request is the caller's problem to fix, so pass the code through.
        return JSONResponse(status_code=exc.status, content={"error": exc.code, "detail": exc.error})
    return decide(ranked)
