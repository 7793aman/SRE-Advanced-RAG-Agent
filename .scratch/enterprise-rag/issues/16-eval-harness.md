# Module 10a — Ragas eval harness

Type: task
Status: open
Blocked by: 15

## Question

The offline evaluation harness that proves each advanced technique beats the naïve baseline.

### Deliverables
- `eval/schema.py` — `Golden` pydantic model + `load_goldens` (YAML, dup-id check).
- `eval/seed_questions.yaml` — the 40 K8s goldens from `_reference/eval/seed_questions.yaml`.
  **Reconcile `golden_sources`**: goldens say `pods.html`; ingested `source` is
  `concepts__workloads__pods.html`. Pick one fix (rename goldens / substring-match in
  `source_overlap` / re-slug ingested source) and apply it. Document the choice.
- `eval/profiles.py` — the 7 flag profiles (`naive`, `sparse_only`, `hybrid`,
  `hybrid+rerank`, `hybrid+rerank+hyde`, `hybrid+rerank+crag`, `all`).
- `eval/invokers.py` — `ServiceInvoker` calling `run_rag_with_trace_no_cache`.
- `eval/ragas_adapter.py` — Ragas `faithfulness / context_precision / context_recall /
  answer_relevancy` with OpenAI LLM + embeddings.
- `eval/post_checks.py` (`forbidden_keywords_check`, `source_overlap`),
  `eval/reporting.py` (`aggregate`, `print_table`), `eval/run_ragas.py` (CLI).
- `Makefile` `eval-*` targets already stubbed in ticket 01 — confirm they run.

### Reference
`_reference/eval/*` (added in commit `96cbbd3` era; present at HEAD).

### Acceptance
- `make eval-baseline` writes `eval/results/<ts>_naive.json`.
- `make eval-all` writes `<ts>_all.json`.
- `make eval-diff` shows `all` ≥ `naive` on context_recall / faithfulness.
