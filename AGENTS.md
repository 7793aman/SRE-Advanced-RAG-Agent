# AGENTS.md — Enterprise RAG

This repo is a **faithful, from-scratch rebuild** of the course project
[`yashprogrammer/EnterpriseRAG_live`](https://github.com/yashprogrammer/EnterpriseRAG_live),
done story-by-story so the owner learns every part rather than copy-pasting the finished code.

- The reference implementation is cloned (read-only, gitignored) at `_reference/`.
- The project report is at `projectReport.pdf`; the reference's own copy is `_reference/PROJECT_REPORT.md`.
- The noise corpus (802 MB, 820 PDFs) is at `noisy_data 2/` in the repo root (gitignored).

## Working agreement

- **Spec → owner codes 100%.** Each wayfinder ticket carries a spec + acceptance criteria +
  a pointer to the reference commit/files. The owner writes all implementation code.
  The agent's job on a ticket is: flesh out the spec, then review the owner's code against
  the reference and the report.
- Build order is fixed by the wayfinder map's blocking edges. Don't skip ahead.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<effort>/`. The wayfinder effort is `.scratch/enterprise-rag/`.
See `docs/agents/issue-tracker.md`.

### Domain docs

single-context. `CONTEXT.md` at the repo root, ADRs under `docs/adr/`.
See `docs/agents/domain.md`.
