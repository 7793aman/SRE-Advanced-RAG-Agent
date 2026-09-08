# AGENTS.md — Enterprise RAG

An enterprise RAG system for Kubernetes IT-operations: a FastAPI service that answers
an SRE's natural-language questions via documentation retrieval, Text2SQL, or both,
behind a layered security pipeline and a multi-tier cache, orchestrated with LangGraph.

## How this project is built

- **The owner is learning this codebase, not just shipping it.** Every session works
  exactly one ticket, then teaches it. After implementing:
  1. Post a comment on the ticket: what was built, key decisions, any deviation from
     `spec.md` or the reference.
  2. In the chat, walk the owner through the implementation — the design of each new
     module, why it's shaped that way, how the pieces connect — and answer follow-up
     questions.
  3. Only then close the ticket. Do not start the next one in the same session.
- Work is tracked as tickets on the configured issue tracker (see `## Agent skills`).
  Pick the lowest-numbered open ticket whose "Blocked by" tickets are all closed.
- Each ticket is a vertical slice with acceptance criteria. Implement it, add tests at
  the seams named in `spec.md`'s Testing Decisions, run `/code-review`, commit.
- `spec.md` is the source of truth for intended behaviour. Tickets deliver its user
  stories; each ticket's `Spec:` line points at the sections that govern it.
- `_reference/` holds a working implementation of a similar system, kept locally as an
  implementation aid the agent may consult. It is gitignored and is not the spec.
- `projectReport.pdf` is the original design brief this system is based on.
- The noise corpus (802 MB) lives at `noisy_data 2/` in the repo root (gitignored).

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in `7793aman/SRE-RAG-Agent`, driven via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, each label string equal to its name (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
