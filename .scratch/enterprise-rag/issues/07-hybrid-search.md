# Module 2 — Hybrid search (dense + sparse + RRF)

Type: task
Status: open
Blocked by: 06

## Question

Add sparse retrieval alongside dense and fuse with Reciprocal Rank Fusion.

### Deliverables
- `app/services/sparse_vector_service.py` — `SparseVectorIndex` (**scikit-learn
  `TfidfVectorizer` + cosine similarity**, not BM25/FastEmbed despite the report),
  thread-safe `fit`/`search`; `fuse_rrf(result_lists, rrf_k=60)`.
- `app/services/vector_store.py` — add `_build_sparse_index` (scroll all Qdrant points →
  fit TF-IDF), `sparse_search(query_text, top_k)`, `hybrid_search(query_embedding,
  query_text, top_k, rrf_k, sparse_top_k)`.
- `app/services/rag_service.py::_retrieve` — branch on `search_mode`: `sparse` →
  `sparse_search`, `hybrid` → embed + `hybrid_search`, else dense.

### Reference
commit `2b68f19` — `app/services/sparse_vector_service.py`, `vector_store.py` diff,
`rag_service.py` diff, `models.py` (`search_mode` added).

### Acceptance
- `search_mode:"sparse"` on an exact-token query (`imagePullPolicy`, `CrashLoopBackOff`)
  retrieves the right K8s doc that `dense` missed.
- `search_mode:"hybrid"` is at least as good as the better of the two on a small set of
  eval goldens.
