# Module 8 — 5-tier caching + doc-dedup storage

Type: task
Status: open
Blocked by: 13

## Question

The cache *service* exists from ticket 05; now wire every expensive call through it,
add the SHA-256 document-dedup storage layer, and surface cache telemetry.

### Deliverables
- `app/storage/storage_backend.py` — `StorageBackend` ABC + `get_storage_backend()` factory.
- `app/storage/local_storage.py` (or `__init__.py` — the reference misplaced `LocalStorage`
  into `__init__.py` and left `local_storage.py` empty; put it where it belongs and note it),
  `app/storage/s3_storage.py` (boto3).
- `app/services/doc_cache_service.py` — `DocCacheService`: `compute_content_hash` /
  `compute_file_hash`, `exists` / `set_metadata` / `get_metadata` keyed by `<hash>/metadata.json`.
- Confirm the cache is consulted at every tier: `embedding` (ticket 05), `intent`
  (`router_service`), `sql_gen` + `sql_result` (`sql_service`), `rag_answer` (`rag_service`).
  Add cache-hit flags to `ChatResponse` / `ResponseMetadata`.
- `app/api/admin.py` — finalize `/admin/cache/stats` (per-tier hits/misses/sets/hit_rate) and
  `/admin/cache/clear`. Optionally wire doc-dedup into `scripts/seed_db.py` so re-ingesting a
  file is a no-op.

### Reference
commits `27171e5`, `99dfa5c`, `b3761be` — `app/storage/*`, `app/services/doc_cache_service.py`,
`app/services/{rag_service,query_cache_service}.py` diffs, `app/api/admin.py` diff.

### Acceptance
- Same `/query` twice → second response `cache_hit: true`, near-zero latency.
- `GET /admin/cache/stats` shows non-zero hits across `embedding` and `rag_answer`.
- Re-running `make seed` on an already-ingested file skips it (if dedup wired).
