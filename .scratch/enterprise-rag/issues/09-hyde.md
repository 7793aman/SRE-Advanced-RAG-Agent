# Module 4 — HyDE (Hypothetical Document Embeddings)

Type: task
Status: open
Blocked by: 08

## Question

Bridge the short-query / long-doc vocabulary gap by retrieving with LLM-generated
hypothetical answers instead of the raw question.

### Deliverables
- `app/services/hyde.py` — `HyDERetriever(num_hypotheses=settings.hyde_num_hypotheses)`:
  generate N hypothetical answers (temp 0.7) + keep the original question, `embed_texts` all,
  `search` with each embedding, dedupe by normalized text keeping the best score, return
  top_k.
- `app/services/rag_service.py::_retrieve` — when `enable_hyde`, use
  `HyDERetriever().retrieve(question, top_k=retrieve_k)` as the retrieval step (before rerank).

### Reference
commit `1d4f8a0` — `app/services/hyde.py`, `rag_service.py` diff.

### Acceptance
- A vague symptom query ("why is my pod getting OOMKilled?") retrieves the relevant
  resource-limits / eviction doc with `enable_hyde:true` but not with it off.
