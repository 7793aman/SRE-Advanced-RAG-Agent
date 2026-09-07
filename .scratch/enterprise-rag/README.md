# How to use this map

**The map:** `map.md` — read it first every session.
**The tickets:** `issues/NN-*.md`, numbered, with a `Blocked by:` line.

## Working a ticket (Phase 2)

1. Read `map.md` (Destination + Notes + Decisions so far).
2. Pick the ticket: the lowest-numbered one whose `Blocked by` tickets are all
   `Status: resolved`, and that isn't already `Status: claimed`. Or the owner names one.
3. Claim it: set `Status: claimed` in the file, save.
4. Expand the spec: read the reference files/commit the ticket names, in detail. Hand the
   owner a concrete implementation spec + acceptance checks.
5. Owner writes the code. Agent reviews it against `_reference/` and `projectReport.pdf`
   (`mattpocock-skills:code-review`).
6. Resolve: append an `## Answer` section (what was built, decisions made, any deviation
   from the reference), set `Status: resolved`, and add a one-line entry to `map.md`'s
   "Decisions so far".

Never resolve more than one ticket per session.

## Frontier right now

Takeable (nothing blocking them): **01 — Project skeleton** and **00 — Prereqs & infra**.
Everything else is blocked until those land.
