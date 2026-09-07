# Pydantic request/response models

Type: task
Status: open
Blocked by: 01

## Question

Build `app/models.py` — every request/response schema the system uses, including the
Layer-1 regex injection validators.

### Deliverables (all in `app/models.py`)
- `ChatRequest` (message, 1–2000 chars, `validate_message_content` field validator with the
  4 injection regex patterns + "must contain actual text" check).
- `QueryRequest` — `question`, `enable_rerank` (default False), `top_k` (1–50, default 5),
  `enable_hyde` (False), `search_mode: Literal["dense","sparse","hybrid"]` (default `"dense"`
  — match the reference exactly even though the README examples say `hybrid`), `enable_crag`
  (True), `enable_self_reflective` (False), plus the same injection-pattern field validator.
- `RetrievedChunk` / `RetrievedChunkPreview` (text, source, score=0.0).
- `ResponseMetadata` (route, retrieved_chunks, cache_hit, reflection_iterations,
  reflection_score, refined_question).
- `PendingSQLBlock` (sql, query_id, explanation).
- `ChatResponse` (answer, sources, confidence 0–1, pending_sql, cache_hit, cost_saved,
  metadata).
- `CRAGEvaluation`, `ReflectionResult`.

### Reference
commit `1d9e264` then final state — `app/models.py`. (Fields accreted across commits;
build the final version now, it's small.)

### Acceptance
- Unit tests (`mattpocock-skills:tdd`): each injection string in the reference's pattern list
  raises `ValidationError`; a normal K8s question passes; `top_k=0` and `top_k=99` rejected.
