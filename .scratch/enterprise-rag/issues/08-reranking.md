# Module 3 — Cross-encoder reranking

Type: task
Status: open
Blocked by: 07

## Question

Re-score the top-N retrieved candidates with a cross-encoder before generation.

### Deliverables
- `app/services/reranking.py` — `Reranker` with pluggable backend from
  `settings.reranker_backend`: `local` (sentence-transformers `CrossEncoder`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) or `voyage` (`voyageai.Client.rerank`).
  Lazy model load; on any failure return the original order truncated.
- `app/services/rag_service.py::_retrieve` — when `enable_rerank`, retrieve
  `settings.reranker_initial_top_k` (20) candidates then `Reranker().rerank(question,
  chunks, top_k=final_top_k)`; else just slice to `final_top_k`.

### Reference
commit `57af24d` — `app/services/reranking.py`, `rag_service.py` diff.

### Acceptance
- On a query where the gold chunk sits at rank ~8 after hybrid retrieval, `enable_rerank:true`
  promotes it into the top 5.
- `RERANKER_BACKEND=local` works offline after first model download.
