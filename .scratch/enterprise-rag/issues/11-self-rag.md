# Module 6 — Self-RAG reflection loop

Type: task
Status: open
Blocked by: 10

## Question

After generation, have a critic LLM score the answer; regenerate with a sharper question
if it's weak (max 2 retries).

### Deliverables
- `app/services/self_reflective.py` — `reflect_on_answer(question, answer, context) ->
  ReflectionResult` (strict 4-criterion rubric prompt, JSON out), `should_regenerate(
  reflection, iteration)` (needs_regeneration AND score < `reflection_min_score` AND
  iteration < `max_reflection_retries`).
- `app/services/rag_service.py::_generate` — when `enable_self_reflective`, loop:
  reflect → if `should_regenerate`, set `working_q = refined_question`, regenerate, count
  iteration. Surface `reflection_iterations` / `reflection_score` / `refined_question` in
  `ResponseMetadata`.

### Reference
commit `3a4cd05` — `app/services/self_reflective.py`, `rag_service.py` diff, `models.py`
(`ReflectionResult`, metadata fields).

### Acceptance
- A vague query ("tell me about scaling") produces a refined question and a second
  generation; `metadata.reflection_iterations >= 1`.
- A crisp query passes reflection on the first try (0 iterations).
