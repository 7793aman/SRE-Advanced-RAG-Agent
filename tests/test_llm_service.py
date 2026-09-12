"""Unit seam: generate_text() / generate_json(), with the OpenAI client faked.

Nothing here touches a real OpenAI call — the fake client records what it was
asked and returns a fixed completion, so we can check the request shape and
the token-usage parsing without spending real API calls.
"""

from __future__ import annotations

import pytest

from app.config import settings


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeCompletion:
    def __init__(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletionsAPI:
    """Records every call it's asked to make; returns a fixed completion."""

    def __init__(self, content: str = "a fixed answer") -> None:
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _FakeCompletion:
        self.calls.append(kwargs)
        return _FakeCompletion(self._content)


class _FakeChatAPI:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletionsAPI(content)


class _FakeOpenAIClient:
    def __init__(self, content: str = "a fixed answer") -> None:
        self.chat = _FakeChatAPI(content)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIClient:
    client = _FakeOpenAIClient()
    monkeypatch.setattr("app.services.llm_service._get_client", lambda: client)
    return client


def test_generate_text_returns_the_model_reply_and_token_usage(
    fake_client: _FakeOpenAIClient,
) -> None:
    from app.services.llm_service import generate_text

    result = generate_text("What is a pod?")

    assert result.text == "a fixed answer"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15


def test_generate_text_sends_the_system_prompt_as_its_own_message(
    fake_client: _FakeOpenAIClient,
) -> None:
    from app.services.llm_service import generate_text

    generate_text("What is a pod?", system_prompt="Be terse.")

    call = fake_client.chat.completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "Be terse."}
    assert call["messages"][1] == {"role": "user", "content": "What is a pod?"}
    assert call["model"] == settings.llm_model_answer


def test_generate_text_without_system_prompt_sends_only_a_user_message(
    fake_client: _FakeOpenAIClient,
) -> None:
    from app.services.llm_service import generate_text

    generate_text("What is a pod?")

    call = fake_client.chat.completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": "What is a pod?"}]


def test_generate_json_requests_json_mode_and_uses_the_grader_model(
    fake_client: _FakeOpenAIClient,
) -> None:
    from app.services.llm_service import generate_json

    generate_json("Classify this question.")

    call = fake_client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == settings.llm_model_grader


def test_generate_text_accepts_a_model_override(fake_client: _FakeOpenAIClient) -> None:
    from app.services.llm_service import generate_text

    generate_text("hi", model="gpt-4o-mini")

    assert fake_client.chat.completions.calls[0]["model"] == "gpt-4o-mini"


def test_generate_text_omits_temperature_by_default(fake_client: _FakeOpenAIClient) -> None:
    # Some models (e.g. gpt-5.6-terra) reject any temperature other than
    # their own default with a 400 — omitting the key entirely when the
    # caller hasn't asked for a specific value avoids that everywhere.
    from app.services.llm_service import generate_text

    generate_text("hi")

    assert "temperature" not in fake_client.chat.completions.calls[0]


def test_generate_text_sends_temperature_only_when_explicitly_given(
    fake_client: _FakeOpenAIClient,
) -> None:
    from app.services.llm_service import generate_text

    generate_text("hi", temperature=0.2)

    assert fake_client.chat.completions.calls[0]["temperature"] == 0.2


def test_generate_json_omits_temperature_by_default(fake_client: _FakeOpenAIClient) -> None:
    from app.services.llm_service import generate_json

    generate_json("Classify this.")

    assert "temperature" not in fake_client.chat.completions.calls[0]
