"""Unit seam: PII redaction (L7a on the question, L7b on the answer, story #33)."""

from __future__ import annotations

import pytest

from app.security.pii_redaction import redact_pii


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Email me at aman@gmail.com please", "Email me at [EMAIL] please"),
        ("Call +1 415 555 0134 tonight", "Call [PHONE] tonight"),
        ("Card 4111 1111 1111 1111 expired", "Card [CARD] expired"),
        ("Server at 10.0.12.7 is down", "Server at [IP] is down"),
    ],
)
def test_each_kind_of_pii_is_replaced_with_a_placeholder(raw: str, expected: str) -> None:
    assert redact_pii(raw) == expected


def test_several_values_in_one_text_are_all_redacted() -> None:
    raw = "Ask ravi@company.com or call +1 415 555 0134"

    assert redact_pii(raw) == "Ask [EMAIL] or call [PHONE]"


def test_ordinary_kubernetes_text_is_not_touched() -> None:
    raw = "Set terminationGracePeriodSeconds=30 and replicas: 3 on kube-system v1.29.4"

    assert redact_pii(raw) == raw


def test_a_contiguous_digit_run_that_fails_the_luhn_check_is_left_alone() -> None:
    # 16 digits with no card formatting and no valid checksum — e.g. a glued-together
    # timestamp/id pair an SRE might paste, not a card number.
    raw = "epoch 1758472800000123456789 in the trace"

    assert redact_pii(raw) == raw


def test_a_spaced_card_number_with_the_wrong_checksum_is_left_alone() -> None:
    raw = "Card 1234 5678 9012 3456 was tried"

    assert redact_pii(raw) == raw
