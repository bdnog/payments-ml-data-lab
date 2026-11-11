import base64
import json
from datetime import date

import pytest

from settlement.storage.cursors import (
    InvalidCursorError,
    decode_break_cursor,
    decode_settlement_cursor,
    encode_break_cursor,
    encode_settlement_cursor,
)

DAY = date(2026, 1, 14)


def forge(payload) -> str:
    raw = json.dumps(payload).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def test_settlement_cursor_round_trips():
    token = encode_settlement_cursor(DAY, "mer_5d7a1e")
    assert decode_settlement_cursor(token) == (DAY, "mer_5d7a1e")


def test_break_cursor_round_trips():
    token = encode_break_cursor(DAY, "brk_5014000a")
    assert decode_break_cursor(token) == (DAY, "brk_5014000a")


def test_tokens_are_opaque_url_safe_and_unpadded():
    token = encode_settlement_cursor(DAY, "mer_5d7a1e")
    assert "=" not in token and "+" not in token and "/" not in token
    assert "mer_" not in token  # base64, not plain text


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not base64 !!",
        "e30",  # {}
        forge([1, 2]),  # not an object
        forge({"d": "2026-01-14"}),  # missing key
        forge({"d": "2026-01-14", "m": "mer_5d7a1e", "x": "1"}),  # extra key
        forge({"d": 20260114, "m": "mer_5d7a1e"}),  # wrong type
        forge({"d": "2026-13-45", "m": "mer_5d7a1e"}),  # impossible date
        forge({"d": "2026-01-14", "m": "'; DROP TABLE"}),  # not a merchant id
        forge({"d": "2026-01-14", "b": "brk_5014000a"}),  # a break cursor on settlements
        "A" * 500,
    ],
)
def test_bad_settlement_cursors_are_rejected(token):
    with pytest.raises(InvalidCursorError):
        decode_settlement_cursor(token)


@pytest.mark.parametrize(
    "token",
    [
        forge({"d": "2026-01-14", "m": "mer_5d7a1e"}),  # a settlement cursor on breaks
        forge({"d": "2026-01-14", "b": "brk_zzzzzzzz"}),
        "%%%",
    ],
)
def test_bad_break_cursors_are_rejected(token):
    with pytest.raises(InvalidCursorError):
        decode_break_cursor(token)


def test_a_flipped_character_is_rejected_or_decodes_to_a_valid_position():
    """No signature: a flipped character may land on another valid position, never on garbage."""
    token = encode_settlement_cursor(DAY, "mer_5d7a1e")
    for i in range(len(token)):
        mutated = token[:i] + ("A" if token[i] != "A" else "B") + token[i + 1 :]
        try:
            day, merchant = decode_settlement_cursor(mutated)
        except InvalidCursorError:
            continue
        assert isinstance(day, date) and merchant.startswith("mer_")
