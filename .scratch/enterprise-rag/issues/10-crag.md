# Module 5 — CRAG (Corrective RAG) + Tavily web fallback

Type: task
Status: open
Blocked by: 09, 00

## Question

Grade retrieved chunks for relevance; when they're too weak, fall back to web search
instead of letting the LLM hallucinate.

### Deliverables
- `app/services/web_search.py` — `search_web(query, max_results=5)` via Tavily; raises
  `ValueError` if `TAVILY_API_KEY` unset; maps results to `RetrievedChunk`.
- `app/services/crag.py` — `grade_chunks` (LLM JSON grader → `CRAGEvaluation`),
  `should_trigger_web_search` (score < `settings.crag_relevance_threshold`),
  `crag_pipeline(question, chunks, enable_crag)` returning `(chunks, evaluation, used_web)`;
  handles the "no chunks at all" case.
- `app/services/rag_service.py::_retrieve` — run `crag_pipeline` after rerank; log the grade.

### Reference
commit `bf3b02e` — `app/services/crag.py`, `web_search.py`, `rag_service.py` diff.

### Acceptance
- "What is the latest stable Kubernetes release?" (not in the frozen corpus) → CRAG grade low
  → Tavily results used → answer cites a URL source.
- An in-corpus question does NOT trigger web fallback.
- With `TAVILY_API_KEY` removed, low-grade path degrades gracefully (no crash).
