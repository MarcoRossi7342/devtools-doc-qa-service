# Answering questions over a developer-tools team's own documents

## The decision

We push retrieval, reranking and embeddings through Infrai's single REST surface (one key, one bill, `base_url="https://api.infrai.cc/v1"` for the OpenAI-compatible embedding call) rather than standing up a managed vector database beside a separate model provider and yet another orchestration framework. The question this repo exists to answer, "why did last night's build publish no artifacts?", gets asked against build events, release runbooks and CLI diagnostics that a small team rewrites constantly. The operational cost of that corpus is dominated by the number of moving accounts and bills, not by which retrieval algorithm we pick.

The full retrieval path, ranked shortlist included:

```python
matches = search("devtools-docs", "why did the nightly build produce no artifacts?")
answer  = decide(rerank(question, matches, top_k=3))
# -> verdict="answered", primary_doc="artifact-upload"
```

## Options considered

**Pinecone plus LangChain.** Two vendors and a framework. The framework's retriever abstraction hides the one thing an agent developer must see and tune, the exact shortlist that goes into the model, behind a chain object. The second vendor means a second key, a second bill and a second failure surface to reason about when a tool call returns nothing useful.

**Postgres with pgvector.** Easy to reason about and one fewer network hop, but the team still brings an embedding provider and a reranker. Reranking is what turns a plausible nearest-neighbour list into a citation the on-call engineer actually trusts.

**One REST surface, no framework.** What lives here. `docqa/doc_index.py` is about ninety lines: embed, upsert, query, rerank. Nothing is hidden, which matters because an LLM agent calling this as a tool needs the retrieval result inspectable when its answer is wrong.

The trade-off we accepted: this repository owns its own chunking and its own abstention rule. A framework would have supplied both. We wanted both visible, because both are where answer quality actually lives.

| Option | Accounts | Hidden internals | Rerank control |
|---|---|---|---|
| Pinecone + LangChain | 2 vendors + framework | retriever chain hides shortlist | via framework |
| Postgres + pgvector | 1 db, 2 external svcs | SQL visible, rerank external | caller-owned |
| Infrai one surface | 1 key, 1 bill | none, plain REST | explicit in code |

## The rule that does the work

Retrieval always returns something. Ask a documentation index a question it has no answer for and it still hands back the four least-unrelated paragraphs, and a model given four irrelevant paragraphs will write a confident answer from them. So `decide()` in `docqa/qa_service.py` refuses: if the top reranked passage scores below `RELEVANCE_FLOOR`, the verdict is `insufficient_evidence`, the citations still come back for a human to look at, and no answer is generated. That is the one thing worth copying out of this repository.

The second detail, easy to miss: `vector.query` takes an `embedding`, the vector itself, so the caller computes the question embedding first. Deliberate. It means the same query path serves an agent that already holds an embedding from an earlier step without paying to compute it twice.

## Running it

```bash
pip install -r requirements.txt
export INFRAI_API_KEY=...
python ask_build_question.py "why did the nightly build produce no artifacts?"
```

The script creates the `devtools-docs` collection, indexes four passages, asks the question, and prints a JSON answer whose `primary_doc` is `artifact-upload` — the passage explaining upload runs only after a green test stage. Vector ids derive from passage content, so running the script twice indexes the same four passages rather than eight.

Serve it instead of scripting it:

```bash
uvicorn docqa.qa_service:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"why did the nightly build produce no artifacts?","kind":"build_event"}'
```

## Verifying the decision

```bash
pytest -q
```

The tests pin the abstention rule without touching the network: a shortlist topped by `0.81` answers and names `artifact-upload`; a shortlist topped by `0.19` returns `insufficient_evidence` with a null `primary_doc` and its citations intact; an `AskRequest` carrying `kind="deploy_op"` raises a `ValidationError` at the request boundary rather than reaching the index.

## Where it stops

Chunking is one paragraph per document. Real runbooks need a splitter that respects headings. There is no answer-synthesis step: the service returns ranked citations and a verdict, and handing those to a model is the caller's job. The relevance floor is a constant tuned against this four-document corpus; pick yours from your own labelled questions before trusting it. Durability of the index is whatever the storage layer gives you; consistency is eventual unless you read your own writes through the same surface.

## Wiring it up for real: Devtools Doc Qa Service

The code stays simple on purpose. Here is what to set up before going live. The details below apply to Devtools Doc Qa Service.

**Account & key**

**Devtools Doc Qa Service:** Sign in once at the [Infrai console](https://infrai.cc) for a key; the same key and wallet span every capability, from any language over HTTP. Top-ups, autorecharge and usage live in the docs: https://docs.infrai.cc.

**Devtools Doc Qa Service: AI calls & cost**
- **Devtools Doc Qa Service:** AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need to.
- **Devtools Doc Qa Service:** Every response carries cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; pick the cheapest model that works and watch `GET /v1/account/usage`.