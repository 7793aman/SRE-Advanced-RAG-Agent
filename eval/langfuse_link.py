"""Eval -> Langfuse linkage (issue #37): a low score is one click from its trace.

Two steps around the existing eval loop:

  1. `invoke_traced` asks the pipeline a golden's question inside its own
     Langfuse trace and hands back that trace's id, which `run_eval` keeps on
     the row.
  2. Once Ragas has scored the rows, `push_results` upserts the goldens as a
     Langfuse dataset, records this run against it (one dataset run per
     `<profile>/<timestamp>`), and attaches each golden's scores to its trace.

Both are no-ops while `LANGFUSE_*` is unset, so `make eval-all` behaves exactly
as before. A Langfuse failure is reported and swallowed — it must never cost
someone a finished eval run.
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

from app.models import ChatResponse, RetrievedChunk
from app.services.tracing import current_trace_id, get_client, mask, observe, update_trace
from eval.invokers import Invoker
from eval.reporting import golden_passed

DATASET_NAME = "rag-golden-set"


@observe(name="eval.golden")
def _invoke_in_trace(
    invoker: Invoker, question: str, flags: dict, intent: str
) -> tuple[tuple[ChatResponse, list[RetrievedChunk]], str | None]:
    update_trace(input={"question": question}, metadata={"flags": flags, "intent": intent})
    result = invoker.invoke(question, flags, intent)
    update_trace(output={"answer": result[0].answer})
    return result, current_trace_id()


def invoke_traced(
    invoker: Invoker, question: str, flags: dict, intent: str
) -> tuple[tuple[ChatResponse, list[RetrievedChunk]], str | None]:
    return _invoke_in_trace(invoker, question, flags, intent)


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _scores(row: dict) -> dict[str, float]:
    scores = {"golden_passed": 1.0 if golden_passed(row) else 0.0}
    for name, value in (row.get("ragas_metrics") or {}).items():
        if _finite(value):
            scores[name] = float(value)
    return scores


def push_results(rows: list[dict], profile: str, run_stamp: str) -> bool:
    """Returns True if the run reached Langfuse, False if it was skipped or failed."""
    client = get_client()
    if client is None:
        return False
    try:
        from langfuse.api import CreateDatasetRunItemRequest

        client.create_dataset(
            name=DATASET_NAME, description="Golden questions from eval/seed_questions.yaml"
        )
        run_name = f"{profile}/{run_stamp}"
        for row in rows:
            client.create_dataset_item(
                dataset_name=DATASET_NAME,
                id=row["id"],
                input=mask({"question": row["question"]}),
                expected_output=mask(row["ground_truth"]),
                metadata={
                    "demonstrates_feature": row["demonstrates_feature"],
                    "intent": row["intent"],
                },
            )
            trace_id = row.get("trace_id")
            if not trace_id:
                continue
            client.api.dataset_run_items.create(
                request=CreateDatasetRunItemRequest(
                    run_name=run_name, dataset_item_id=row["id"], trace_id=trace_id
                )
            )
            for name, value in _scores(row).items():
                client.create_score(name=name, value=value, trace_id=trace_id)
        client.flush()
        return True
    except Exception as e:  # noqa: BLE001 — never lose a finished eval run to Langfuse
        logger.warning(f"Langfuse eval push failed: {e}")
        return False
