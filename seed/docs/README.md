# Seed document corpus

The knowledge base is deliberately **~5% signal / ~95% noise** so every advanced
retrieval technique has to earn its place by rescuing signal from noise.

| Directory | What | Committed? |
|-----------|------|------------|
| `true_data/` | 47 real Kubernetes documentation files (concepts, tasks, tutorials, API reference). **Signal** — always ingested in full. | Yes |
| `noisy_data/` | ~820 unrelated technical PDFs (papers on hashing, networking, databases, …). **Noise** — sampled by `--noise-sample`. | Bodies gitignored; only `.gitkeep` is tracked |

## Corpus wiring

The seeder (`scripts/seed_db.py`) reads `noisy_data/` **directly** — no symlink,
no config indirection. The noise bodies are staged there locally and excluded
from git via `.gitignore` (`seed/docs/noisy_data/*`, keeping `.gitkeep`). A repo
root `noisy_data 2/` staging folder, if present, is also gitignored; copy or move
its contents into `seed/docs/noisy_data/` before seeding.

## Selecting the corpus

```bash
uv run python scripts/seed_db.py --no-ingest          # DB only
uv run python scripts/seed_db.py --noise-sample 150   # 47 signal + 150 noise (default)
uv run python scripts/seed_db.py --noise-sample all   # 47 signal + every noise file
```

Noise sampling uses a fixed seed, so `--noise-sample N` always picks the same N
files. Signal is never sampled.
