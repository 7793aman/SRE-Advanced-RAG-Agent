# Data: K8s ops schema & DB seeding

Type: task
Status: open
Blocked by: 01, 03

## Question

Get the operational SQL data and the seeding orchestrator in place.

### Deliverables
- `seed/migrations/003_seed_k8s_ops.sql` — copy from `_reference/seed/migrations/003_seed_k8s_ops.sql`
  verbatim (7 tables: clusters, nodes, deployments, pods, incidents, alerts, oncall_logs +
  ~10k synthetic rows). **Note the real schema differs from the report** — trust the file
  (`environment` not `provider`, `alertname`, `rca_summary`, no `node_count`).
- `scripts/seed_db.py` — `run_migrations` (runs every `.sql` in `seed/migrations/` in order),
  `seed_users` (the two demo users: `agent@demo.local`/`agent123`, `admin@demo.local`/`admin123`),
  `seed_docs` orchestrator + `_ingest_one` + `_select_corpus` (true_data always full, noisy_data
  sampled with a fixed seed; `--noise-sample N|all`, `--no-ingest`). Doc ingestion itself
  depends on ticket 05 — until then `make seed` runs with `--no-ingest`.
- **Noise corpus decision:** wire `noisy_data 2/` (repo root, 820 files) to where
  `_select_corpus` looks (`seed/docs/noisy_data/`). Symlink, copy, or add a config override —
  decide and document in the resolution.
- `seed/docs/true_data/` (47 files) is **already staged** in this repo (done during charting).
  `seed/docs/noisy_data/.gitkeep` exists. Just verify they're present.

### Reference
commit `1d9e264` — `scripts/seed_db.py`, `seed/migrations/`.

### Acceptance
- `uv run python scripts/seed_db.py --no-ingest` creates all 8 tables + 2 users.
- `psql ... -c 'select count(*) from incidents;'` returns a non-trivial number.
- `_select_corpus` returns 47 true + N noisy paths.
