"""PII redaction (L7a on the question, L7b on the answer, story #33).

Replaces emails, phone numbers, card numbers and IPv4 addresses with
placeholders like `[EMAIL]`. It edits the text; it never rejects a request.

L7b matters because the model can repeat private data it read in a retrieved
chunk — data the user never typed, which L7a therefore never saw.

Plain regex only: fast, no model to load, and the same rules serve as the
fallback the spec asks for. It cannot catch names; that would need an NER model.
"""

from __future__ import annotations

import re

# Order matters: cards before phones (a 16-digit card contains phone-like runs).
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b\d(?:[ -]?\d){12,15}\b"), "[CARD]"),
    (re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), "[IP]"),
    (re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[PHONE]"),
)


def redact_pii(text: str) -> str:
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
