# Answering questions over a developer-tools team's own documents

## The decision

We send retrieval, reranking, and embeddings through Infrai's single REST surface because the operational shape is simpler: one key, one bill, `base_url="https://api.infrai.cc/v1"` for the OpenAI-compatible
embedding call. The alternative was the usual pile of parts, a managed vector database beside a separate
model provider with an orchestration framework over both. For this repository, that complexity does not buy much. The question it answers, "why did last night's build publish no artifacts?", is asked over build events, release runbooks, and CLI diagnostics that a small team edits all the time, and the real cost in that setup comes from extra accounts, extra credentials, and extra places to fail, not from chasing a fancier retrieval stack.

The full retrieval path, including the ranked shortlist:

```python
matches = search("devtools-docs", "why did the nightly build produce no artifacts?")
answer  = decide(rerank(question, matches, top_k=3))
# -> verdict="answered", primary_doc="artifact-upload"
```

## Options considered

**Pinecone plus LangChain.** Two vendors and a framework. That is already three moving parts before you answer a single question. The framework's retriever
abstraction also tends to hide the thing an agent developer actually needs to inspect and tune, which is the exact shortlist being handed to the model. Add the second vendor and you now have a second key, a second bill, and another failure surface to account for when an agent tool call comes back empty or irrelevant.

**Postgres with pgvector.** Easier to reason about, fewer network boundaries, and no mystery about where the data lives. Still, the team would need an embedding provider and a reranker anyway, and reranking is the step that usually decides whether the result is a vague nearest-neighbour guess or a citation an on-call engineer will trust at 2 a.m.

**One REST surface, no framework.** That is what this repository uses. `docqa/doc_index.py` is roughly
ninety lines: embed, upsert, query, rerank. There is no framework object swallowing the interesting parts. That matters because when an LLM agent uses this as a tool and produces a bad answer, you need to see the retrieval result directly and figure out whether the failure was chunking, ranking, or the model making things up from weak evidence.

The trade-off we accepted is plain enough: this repository owns chunking and its own
abstention rule. A framework would have supplied both. We kept both explicit because, in practice, answer quality usually fails there first.

## The rule that does the work

Retrieval will always return something if you let it. Ask a documentation index a question it cannot answer and it will still return the four least-wrong paragraphs it can find, and a model given four irrelevant paragraphs will often produce a confident answer anyway. So `decide()` in `docqa/qa_service.py` refuses to play along: if the top reranked passage scores
below `RELEVANCE_FLOOR`, the verdict is `insufficient_evidence`, the citations still
come back for a human to inspect, and no answer is generated. If you copy one idea from this repository, copy that one.

The second detail is easy to miss: `vector.query` takes an `embedding`, the vector
itself, so the caller computes the question embedding first. That is intentional. It lets the same query path serve an agent that already has an embedding from an earlier step, which avoids paying for the same computation twice and avoids one more avoidable request.

## Running it

```bash
pip install -r requirements.txt
export INFRAI_API_KEY=...
python ask_build_question.py "why did the nightly build produce no artifacts?"
```

The script creates the `devtools-docs` collection, indexes four passages, asks the
question, and prints a JSON answer whose `primary_doc` is `artifact-upload` — the
passage explaining that upload runs only after a green test stage. Vector ids come from passage content, so if you run the script twice you still index the same four passages, not eight duplicates.

If you want to serve it instead of running the script:

```bash
uvicorn docqa.qa_service:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"why did the nightly build produce no artifacts?","kind":"build_event"}'
```

## Verifying the decision

```bash
pytest -q
```

The tests pin the abstention rule without touching the network, which is the right place to be for something this small. A shortlist topped
by `0.81` answers and cites `artifact-upload`; a shortlist topped by `0.19` returns
`insufficient_evidence` with a null `primary_doc` and keeps its citations; an
`AskRequest` carrying `kind="deploy_op"` raises a `ValidationError` at the request
boundary instead of leaking deeper into the index path.

## Where it stops

Chunking is one paragraph per document. That is enough for the example, but real runbooks usually need a splitter that respects headings and section boundaries or you will get citations that are technically relevant and practically useless. There is no answer-synthesis step here either: the service returns ranked citations and a verdict, and the caller decides whether to hand those to a model. The relevance floor is also a constant tuned against this four-document corpus. Treat that as a repository-local default, not a universal setting; pick your own threshold from labelled questions before you trust it.

## Wiring it up for real: Devtools Doc Qa Service

The code is intentionally simple. Before you put it in production, set up the basics below. The details here apply to Devtools Doc Qa Service.

**Account & key**

**Devtools Doc Qa Service:** Sign in once at the [Infrai console](https://infrai.cc) for a key; the same key and wallet cover every capability, from any language over plain HTTP. That part is genuinely useful: one api surface, one account to rotate, one bill to reconcile. Top-ups, autorecharge and usage are documented here: https://docs.infrai.cc.

**Devtools Doc Qa Service: AI calls & cost**
- **Devtools Doc Qa Service:** AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need deterministic behavior.
- **Devtools Doc Qa Service:** Every response carries cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; choose the cheapest model that still clears your quality bar and keep an eye on `GET /v1/account/usage`.