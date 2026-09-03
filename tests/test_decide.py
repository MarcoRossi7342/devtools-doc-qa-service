from docqa.doc_index import DocChunk
from docqa.qa_service import AskRequest, decide


def ranked(*pairs):
    return [
        {"metadata": {"doc_id": doc_id, "kind": "build_event", "text": f"passage from {doc_id}"},
         "relevance": score}
        for doc_id, score in pairs
    ]


def test_answers_when_top_passage_clears_the_floor():
    answer = decide(ranked(("artifact-upload", 0.81), ("build-cache-policy", 0.22)))
    assert answer.verdict == "answered"
    assert answer.primary_doc == "artifact-upload"
    assert [c.relevance for c in answer.citations] == [0.81, 0.22]


def test_abstains_when_nothing_is_relevant_enough():
    answer = decide(ranked(("release-freeze", 0.19), ("cli-diagnostics", 0.11)))
    assert answer.verdict == "insufficient_evidence"
    assert answer.primary_doc is None
    assert len(answer.citations) == 2


def test_empty_shortlist_abstains():
    assert decide([]).verdict == "insufficient_evidence"


def test_request_model_rejects_an_unknown_document_kind():
    import pytest
    from pydantic import ValidationError

    AskRequest(question="why did the nightly build fail?", kind="release_op")
    with pytest.raises(ValidationError):
        AskRequest(question="why did the nightly build fail?", kind="deploy_op")


def test_vector_id_is_stable_for_the_same_passage():
    chunk = DocChunk("artifact-upload", "build_event", "upload runs last")
    assert chunk.vector_id() == DocChunk("artifact-upload", "build_event", "upload runs last").vector_id()
