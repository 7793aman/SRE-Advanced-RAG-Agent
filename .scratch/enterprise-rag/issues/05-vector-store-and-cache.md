# Embeddings, vector store, doc processing & the cache service

Type: task
Status: open
Blocked by: 01, 02, 00

## Question

The retrieval substrate: the 5-tier cache service (built now because everything leans on it),
OpenAI embeddings, Qdrant dense storage/search, and Docling document parsing.

### Deliverables
- `app/services/query_cache_service.py` — the full `QueryCacheService`: tiers
  `intent / rag_answer / sql_gen / sql_result / embedding`; SHA-256 keys; Upstash Redis with
  in-memory fallback when unconfigured; per-tier hit/miss/set stats; `get_/set_` methods for
  each tier; `stats()`, `clear()`. Module-level `query_cache = QueryCacheService()`.
- `app/services/embedding_service.py` — `embed_texts(texts, model=None)` with per-text
  embedding-cache lookup, batched OpenAI call for misses, cache write-back.
- `app/services/vector_store.py` — `get_client`, `ensure_collection` (size 1536, COSINE),
  `upsert_chunks(chunks, embeddings)`, `search(query_embedding, top_k)` returning
  `RetrievedChunk`s. (Sparse/hybrid functions come in ticket 07 — leave stubs or omit.)
- `app/services/document_processor.py` — `DocumentProcessor` wrapping Docling
  `DocumentConverter` + `HybridChunker`; `process_document(path) -> list[dict]` with
  `text` / `source` / optional `page_number`.

### Reference
commit `1d9e264` — the four files above.

### Acceptance
- Ingest one true_data PDF: `process_document` → chunks → `embed_texts` → `upsert_chunks`;
  then `search(embed_texts(["what is a pod"])[0])` returns a chunk from that file.
- Second identical `embed_texts` call is an embedding-cache hit (`query_cache.stats()`).
