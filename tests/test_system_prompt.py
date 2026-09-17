"""The hardened system prompt carries the guardrails the spec calls for
(story #34): untrusted user/document content, no role changes, no prompt
disclosure. This is a plain string, so the test just asserts those guardrail
ideas are actually present in the text — a cheap tripwire against someone
editing the constant and quietly dropping a rule.

`GENERAL_KNOWLEDGE_SYSTEM_PROMPT` (Self-RAG's skip-retrieval path, ticket #28)
must keep the same security guardrails while dropping the "context-only" rule
that would otherwise make it refuse every general-knowledge question.
"""

from app.security.system_prompt import GENERAL_KNOWLEDGE_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_forbids_revealing_or_discussing_the_prompt() -> None:
    assert "never reveal" in SYSTEM_PROMPT.lower()


def test_forbids_role_or_persona_changes() -> None:
    assert "never change your role" in SYSTEM_PROMPT.lower()


def test_marks_retrieved_content_as_untrusted() -> None:
    assert "untrusted" in SYSTEM_PROMPT.lower()


def test_instructs_answering_only_from_context() -> None:
    assert "only" in SYSTEM_PROMPT.lower()
    assert "context" in SYSTEM_PROMPT.lower()


def test_general_knowledge_prompt_still_forbids_revealing_the_prompt() -> None:
    assert "never reveal" in GENERAL_KNOWLEDGE_SYSTEM_PROMPT.lower()


def test_general_knowledge_prompt_still_forbids_role_changes() -> None:
    assert "never change your role" in GENERAL_KNOWLEDGE_SYSTEM_PROMPT.lower()


def test_general_knowledge_prompt_does_not_demand_context_only_answers() -> None:
    """The whole point of this prompt: it must NOT tell the model to refuse
    when there's no retrieved context, unlike SYSTEM_PROMPT."""
    assert "only from the retrieved context" not in GENERAL_KNOWLEDGE_SYSTEM_PROMPT.lower()
