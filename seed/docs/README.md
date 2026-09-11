# Seed document corpus

The knowledge base is deliberately **~5% signal / ~95% noise** so every advanced
retrieval technique has to earn its place by rescuing signal from noise.

| Directory | What | Committed? |
|-----------|------|------------|
| `true_data/` | 47 real Kubernetes documentation files (concepts, tasks, tutorials, API reference). **Signal** — always ingested in full. | Yes |
| `noisy_data/` | ~820 unrelated technical PDFs (papers on hashing, networking, databases, …). **Noise** — sampled by `--noise-sample`. | Bodies gitignored; only `.gitkeep` is tracked |

## Corpus wiring

The seeder always reads noise from `seed/docs/noisy_data/`. If that directory
is empty, `scripts.seed_db.stage_noise_corpus()` runs automatically before
ingestion and **symlinks** (never copies — the corpus is ~800MB) every file in
from the first repo-root staging folder it finds: `noisy_data 2/`, then
`noisy_data/`. Both are gitignored, so a fresh drop of the corpus at the repo
root is picked up with no manual step.

## Selecting the corpus

```bash
uv run python scripts/seed_db.py --no-ingest          # DB only
uv run python scripts/seed_db.py --noise-sample 150   # 47 signal + 150 noise (default)
uv run python scripts/seed_db.py --noise-sample all   # 47 signal + every noise file
```

Noise sampling uses a fixed seed, so `--noise-sample N` always picks the same N
files. Signal is never sampled.
