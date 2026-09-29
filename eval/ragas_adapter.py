"""Wires this project's rows into Ragas's LLM-as-judge metrics (issue #33, story
56). Ragas never sees the golden YAML or its field names — only the 4 fields
below, built fresh from what the real pipeline returned plus a keyword-based
`reference` stand-in. See `learning/lessons/0010-one-golden-end-to-end.html` for
the full worked walkthrough of this handoff.
"""

from __future__ import annotations

import os

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings

METRICS = [faithfulness, context_precision, context_recall, answer_relevancy]


def _get_ragas_llm():
    """Ragas-compatible judge LLM, using this project's own OpenAI credentials.

    Assigned directly, not `setdefault` — a developer with a different
    `OPENAI_API_KEY` already exported in their shell would otherwise silently
    have this worktree's `.env` key ignored, same as every other OpenAI client
    construction in this codebase (`llm_service.py`, `embedding_service.py`),
    which pass `api_key=settings.openai_api_key` explicitly for the same reason.
    """
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    return llm_factory(settings.llm_model_grader)


def _get_ragas_embeddings():
    """Ragas-compatible embeddings, using this project's own embedding model."""
    lc_emb = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return LangchainEmbeddingsWrapper(lc_emb)


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )


def run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")
