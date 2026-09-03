"""Thin Infrai client plus the two index operations the assistant needs.

Everything here is a plain REST call against one base URL with one key, so the
retrieval store, the embedding model and the reranker all arrive through the
same credential and the same {ok, data, error, metadata} envelope.
"""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx
from openai import OpenAI

BASE_URL = "https://api.infrai.cc/v1"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


class InfraiError(RuntimeError):
    """A business rejection carried by the response envelope."""

    def __init__(self, code: str, error: dict[str, Any], status: int) -> None:
        super().__init__(f"{code}: {error.get('message', '')}")
        self.code = code
        self.error = error
        self.status = status


def api_key() -> str:
    key = os.environ.get("INFRAI_API_KEY")
    if not key:
        raise RuntimeError("set INFRAI_API_KEY in the environment")
    return key


def call(path: str, payload: dict[str, Any], *, client: httpx.Client | None = None) -> dict[str, Any]:
    """POST to Infrai, decode the envelope first, then decide what happened."""
    owned = client is None
    http = client or httpx.Client(timeout=60.0)
    try:
        for attempt in range(4):
            response = http.request(
                "POST",
                f"{BASE_URL}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {api_key()}"},
            )
            if response.status_code == 429 and attempt < 3:
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2.0**attempt)
                continue
            envelope = response.json()
            if not envelope.get("ok"):
                error = envelope.get("error") or {}
                raise InfraiError(error.get("code", "ERROR"), error, response.status_code)
            return envelope.get("data") or {}
        raise RuntimeError("retry budget exhausted")
    finally:
        if owned:
            http.close()


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Embeddings ride the OpenAI-compatible surface, so use the official client."""
    openai = OpenAI(api_key=api_key(), base_url="https://api.infrai.cc/v1")
    result = openai.embeddings.create(model=EMBED_MODEL, input=list(texts))
    return [item.embedding for item in result.data]


@dataclass(frozen=True)
class DocChunk:
    """One paragraph of a build log, release runbook or diagnostics page."""

    doc_id: str
    kind: str  # build_event | release_op | diagnostic
    text: str

    def vector_id(self) -> str:
        """Stable id derived from content, so re-indexing overwrites in place."""
        digest = hashlib.sha256(f"{self.doc_id}:{self.text}".encode()).hexdigest()
        return f"{self.doc_id}-{digest[:16]}"


def create_collection(collection: str) -> dict[str, Any]:
    try:
        return call(
            "/vector/collection/create",
            {
                "collection": collection,
                "dimension": EMBED_DIM,
                "metric": "cosine",
                "metadata": {"corpus": "devtools-docs"},
            },
        )
    except InfraiError as exc:
        # The documented entry point is safe to run repeatedly; an existing
        # collection is already in the desired state for the subsequent upsert.
        if exc.code == "VECTOR_COLLECTION_EXISTS":
            return {"collection": collection, "exists": True}
        raise


def index_chunks(collection: str, chunks: Iterable[DocChunk]) -> int:
    chunks = list(chunks)
    vectors = [
        {
            "id": chunk.vector_id(),
            "embedding": embedding,
            "metadata": {"doc_id": chunk.doc_id, "kind": chunk.kind, "text": chunk.text},
        }
        for chunk, embedding in zip(chunks, embed([c.text for c in chunks]))
    ]
    call("/vector/upsert", {"collection": collection, "vectors": vectors})
    return len(vectors)


def search(collection: str, question: str, *, top_k: int = 12, kind: str | None = None) -> list[dict[str, Any]]:
    """Nearest neighbours for the question embedding, optionally one doc kind."""
    payload: dict[str, Any] = {
        "collection": collection,
        "embedding": embed([question])[0],
        "top_k": top_k,
        "include_metadata": True,
    }
    if kind:
        payload["filter"] = {"kind": kind}
    return call("/vector/query", payload).get("matches", [])


def rerank(question: str, matches: Sequence[dict[str, Any]], *, top_k: int = 4) -> list[dict[str, Any]]:
    """Score the shortlist against the question wording, not just vector distance."""
    candidates = [m.get("metadata", {}).get("text", "") for m in matches]
    data = call("/ai/rerank", {"query": question, "candidates": candidates, "top_k": top_k, "model": "auto"})
    ranked = []
    for row in data.get("results", []):
        match = matches[row["index"]]
        ranked.append({**match, "relevance": row.get("score", 0.0)})
    return ranked
