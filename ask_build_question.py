"""Runnable path: index a handful of toolchain documents, then ask one question.

    export INFRAI_API_KEY=...   # $2 of sign-up credit is enough to run this
    python ask_build_question.py "why did the nightly build stop uploading artifacts?"
"""

from __future__ import annotations

import json
import sys

from docqa.doc_index import DocChunk, create_collection, index_chunks, rerank, search
from docqa.qa_service import DEFAULT_COLLECTION, decide

CORPUS = [
    DocChunk(
        "build-cache-policy",
        "build_event",
        "Nightly builds reuse the remote cache keyed by lockfile hash; a lockfile "
        "change invalidates the key and the build recompiles from scratch, which is "
        "why the first nightly after a dependency bump takes roughly twice as long.",
    ),
    DocChunk(
        "artifact-upload",
        "build_event",
        "Artifact upload runs as the final build step and only executes when the "
        "test stage exits zero, so a failing test leaves the run green in the log "
        "viewer but publishes no artifacts.",
    ),
    DocChunk(
        "release-freeze",
        "release_op",
        "During a release freeze the release bot rejects merges to the release "
        "branch and holds them in the queue until the freeze window closes.",
    ),
    DocChunk(
        "cli-diagnostics",
        "diagnostic",
        "The CLI writes a diagnostics bundle to .devtools/diag containing the "
        "resolved config, the toolchain version and the last hundred cache lookups.",
    ),
]


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "why did the nightly build produce no artifacts?"
    create_collection(DEFAULT_COLLECTION)
    count = index_chunks(DEFAULT_COLLECTION, CORPUS)
    print(f"indexed {count} passages into {DEFAULT_COLLECTION}")

    matches = search(DEFAULT_COLLECTION, question)
    answer = decide(rerank(question, matches, top_k=3))
    print(json.dumps(answer.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
