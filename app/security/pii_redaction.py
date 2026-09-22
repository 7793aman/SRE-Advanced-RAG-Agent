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

# Any 13-16 digit run, spaced or not, is a *candidate* card number — checked
# against the Luhn checksum before being redacted (see `_redact_card`), so an
# ordinary long number an SRE pastes (a timestamp, a resource generation id)
# isn't swallowed just because it happens to be the right length.
_CARD_CANDIDATE = re.compile(r"\b\d(?:[ -]?\d){12,15}\b")

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), "[IP]"),
    (re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[PHONE]"),
)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    digits = re.sub(r"[ -]", "", match.group())
    return "[CARD]" if _luhn_valid(digits) else match.group()


def redact_pii(text: str) -> str:
    text = _CARD_CANDIDATE.sub(_redact_card, text)  # card first: it contains phone-like runs
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
