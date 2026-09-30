"""Thin OpenAI wrapper: plain text generation and JSON-mode generation.

Both return an `LLMResponse` (text plus prompt/completion/total token counts)
so callers can log or budget spend without repeating the OpenAI
response-parsing dance. `generate_json` differs only in asking the model for
a JSON object back; callers `json.loads()` the text themselves since each
caller wants a different shape.
"""

from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import settings
from app.services.lazy_singleton import LazySingleton
from app.services.tracing import observe, update_generation


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


_client: LazySingleton[OpenAI] = LazySingleton(lambda: OpenAI(api_key=settings.openai_api_key))


def _get_client() -> OpenAI:
    return _client.get()


def _messages(prompt: str, system_prompt: str | None) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _to_response(
    completion: ChatCompletion, model: str, messages: list[ChatCompletionMessageParam]
) -> LLMResponse:
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    response = LLMResponse(
        text=text,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )
    # Langfuse prices the call from the model name and these token counts.
    update_generation(
        model=model,
        usage={
            "input": response.prompt_tokens,
            "output": response.completion_tokens,
            "total": response.total_tokens,
        },
        input=messages,
        output=text,
    )
    return response


@observe(name="llm.generate_text", as_type="generation")
def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    model_name = model or settings.llm_model_answer
    messages = _messages(prompt, system_prompt)
    # Not every model accepts a custom temperature — some (gpt-5.6-terra was
    # the one that first surfaced this, with a real 400) only support their
    # own default and reject any other value. Omitting the parameter
    # entirely when the caller hasn't asked for a specific value sidesteps
    # that everywhere, rather than hardcoding a per-model allow-list.
    if temperature is None:
        completion = _get_client().chat.completions.create(model=model_name, messages=messages)
    else:
        completion = _get_client().chat.completions.create(
            model=model_name, messages=messages, temperature=temperature
        )
    return _to_response(completion, model_name, messages)


@observe(name="llm.generate_json", as_type="generation")
def generate_json(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    model_name = model or settings.llm_model_grader
    messages = _messages(prompt, system_prompt)
    if temperature is None:
        completion = _get_client().chat.completions.create(
            model=model_name, messages=messages, response_format={"type": "json_object"}
        )
    else:
        completion = _get_client().chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    return _to_response(completion, model_name, messages)
