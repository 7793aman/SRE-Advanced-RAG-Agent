"""Thin OpenAI wrapper: plain text generation and JSON-mode generation.

Both return an `LLMResponse` (text plus prompt/completion/total token counts)
so callers can log or budget spend without repeating the OpenAI
response-parsing dance. `generate_json` differs only in asking the model for
a JSON object back; callers `json.loads()` the text themselves since each
caller wants a different shape.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import settings


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _messages(prompt: str, system_prompt: str | None) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _to_response(completion: ChatCompletion) -> LLMResponse:
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    return LLMResponse(
        text=text,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )


def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> LLMResponse:
    completion = _get_client().chat.completions.create(
        model=model or settings.llm_model_answer,
        messages=_messages(prompt, system_prompt),
        temperature=temperature,
    )
    return _to_response(completion)


def generate_json(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
) -> LLMResponse:
    completion = _get_client().chat.completions.create(
        model=model or settings.llm_model_grader,
        messages=_messages(prompt, system_prompt),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _to_response(completion)
