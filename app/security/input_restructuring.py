"""Input restructuring (L5, story #31): cap the question at a token ceiling.

An over-long question is cut to `max_input_tokens` before it reaches the guard
model or OpenAI. That bounds cost, stops "stuff 100k tokens of attack text"
tricks, and keeps the L2 guard inside its own input limit. We truncate (keep
the start, drop the tail) rather than summarise: summarising costs an LLM call
on input we haven't yet vetted.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from app.config import settings


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def restructure_input(text: str, max_tokens: int | None = None) -> str:
    limit = settings.max_input_tokens if max_tokens is None else max_tokens
    tokens = _encoding().encode(text)
    if len(tokens) <= limit:
        return text
    return _encoding().decode(tokens[:limit])
