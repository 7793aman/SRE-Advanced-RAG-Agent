# Project notes for Claude

## New worktree setup: copy `.env`

`.env` holds real secrets (`OPENAI_API_KEY`, DB URLs, etc.) and is gitignored
on purpose. That means `git worktree add` does **not** bring it into a new
worktree — a fresh worktree only has `.env.example`, with no real credentials.

When starting work in a new worktree of this repo, check whether `.env`
exists at the worktree root. If it's missing, copy it from the main checkout:

```
cp /Users/aman/dev/RAG/.env .env
```

Without this, anything that calls OpenAI (embeddings, LLM generation) fails
with `openai.OpenAIError: Missing credentials.`
