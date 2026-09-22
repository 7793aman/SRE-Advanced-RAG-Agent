"""llm-guard wrappers: the input scan (L2) and output moderation (half of L7b).

* `scan_input`      — PromptInjection, Toxicity and (optionally) BanTopics on the
  question. Any failure is a `400` with a code naming which scanner fired
  (`injection_blocked`, `toxic_input`, `banned_topic`). Stories #32.
* `moderate_output` — Toxicity (and BanTopics) on the model's answer → `400
  output_blocked`. PII on the answer is handled separately by `pii_redaction`.

The scanners are ML models loaded once, lazily, from Hugging Face. When the
library or a model is unavailable — or a scanner crashes mid-request — the input
scan falls back to regex so there is always *some* guard (spec: "regex fallbacks
when the library or a model is unavailable"). The output scan fails open in that
case and logs, since PII redaction still runs after it.

A failed load is *not* cached: a transient problem (a network blip fetching the
model on first use) would otherwise pin the process to the regex fallback for
its whole lifetime. Only a successful build is cached, so the next request
retries.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from fastapi import HTTPException, status
from loguru import logger

from app.config import settings

# (name, is_valid(text)) — the shape both builders return, so tests can fake it.
_Check = tuple[str, Callable[[str], bool]]

_INPUT_ERRORS = {
    "injection": "injection_blocked",
    "toxicity": "toxic_input",
    "topics": "banned_topic",
}

# Broader than L1's patterns: L1 already ran, so this only has to catch what a
# model would have. Deliberately loose on wording, tight on intent.
_FALLBACK_INJECTION = (
    re.compile(
        r"(?i)\b(?:disregard|set\s+aside|bypass|circumvent)\b.{0,40}\b(?:instructions?|guidance|rules|prompt|guidelines)\b"
    ),
    re.compile(r"(?i)\b(?:pretend|act)\s+(?:to\s+be|as\s+if\s+you\s+are|you\s+are)\b"),
    re.compile(r"(?i)\b(?:developer|jailbreak|dan)\s+mode\b"),
    re.compile(
        r"(?i)\b(?:print|repeat|output|reveal)\b.{0,30}\b(?:hidden|secret|initial)\b.{0,20}\b(?:setup|instructions?|prompt|text)\b"
    ),
)


class _RetryingCache[T]:
    """Caches the result of a `build()` passed to `.get()`, only on success.

    A raised exception logs `unavailable_msg` and returns `None`, uncached — the
    next `.get()` calls `build()` again instead of being stuck with the failure.
    `build` is passed in on every call, not stored, so it's resolved fresh each
    time (lets a caller swap the builder, e.g. in tests).
    """

    def __init__(self, unavailable_msg: str) -> None:
        self._unavailable_msg = unavailable_msg
        self._cached: T | None = None

    def get(self, build: Callable[[], T]) -> T | None:
        if self._cached is not None:
            return self._cached
        try:
            self._cached = build()
        except Exception:
            logger.warning(self._unavailable_msg)
            return None
        return self._cached


def _as_input_check(scanner) -> Callable[[str], bool]:  # noqa: ANN001 - llm-guard has no shared type
    return lambda text: bool(scanner.scan(text)[1])


def _as_output_check(scanner) -> Callable[[str], bool]:  # noqa: ANN001
    return lambda text: bool(scanner.scan("", text)[1])


def _build_input_scanners() -> list[_Check]:
    from llm_guard.input_scanners import BanTopics, PromptInjection, Toxicity

    checks: list[_Check] = [
        (
            "injection",
            _as_input_check(PromptInjection(threshold=settings.prompt_injection_threshold)),
        ),
        ("toxicity", _as_input_check(Toxicity(threshold=settings.toxicity_threshold))),
    ]
    if settings.banned_topics:
        checks.append(("topics", _as_input_check(BanTopics(topics=settings.banned_topics))))
    return checks


def _build_output_scanners() -> list[_Check]:
    from llm_guard.output_scanners import BanTopics, Toxicity

    checks: list[_Check] = [
        ("toxicity", _as_output_check(Toxicity(threshold=settings.output_toxicity_threshold))),
    ]
    if settings.banned_topics:
        checks.append(("topics", _as_output_check(BanTopics(topics=settings.banned_topics))))
    return checks


_input_scanner_cache: _RetryingCache[list[_Check]] = _RetryingCache(
    "llm-guard input scanners unavailable; using regex fallback"
)
_output_scanner_cache: _RetryingCache[list[_Check]] = _RetryingCache(
    "llm-guard output scanners unavailable; skipping output moderation"
)


def _load_input_scanners() -> list[_Check] | None:
    return _input_scanner_cache.get(_build_input_scanners)


def _load_output_scanners() -> list[_Check] | None:
    return _output_scanner_cache.get(_build_output_scanners)


def _blocked(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _regex_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FALLBACK_INJECTION)


def scan_input(text: str) -> None:
    scanners = _load_input_scanners()
    if scanners is not None:
        try:
            for name, is_valid in scanners:
                if not is_valid(text):
                    raise _blocked(_INPUT_ERRORS[name])
            return
        except HTTPException:
            raise
        except Exception:
            logger.exception("llm-guard input scan crashed; using regex fallback")
    if _regex_injection(text):
        raise _blocked(_INPUT_ERRORS["injection"])


def moderate_output(text: str) -> None:
    scanners = _load_output_scanners()
    if scanners is None:
        return
    try:
        for _name, is_valid in scanners:
            if not is_valid(text):
                raise _blocked("output_blocked")
    except HTTPException:
        raise
    except Exception:
        logger.exception("llm-guard output scan crashed; skipping output moderation")
