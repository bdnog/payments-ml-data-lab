"""Opaque keyset cursors: base64url of a small JSON object holding the last key of a page.

Keyset ("seek") pagination continues from the last row seen (``WHERE key > last``) instead of
skipping ``OFFSET n`` rows, so a page costs the same however deep it is and rows are neither
repeated nor skipped when data is added. The token is opaque to clients but not secret or signed: it
is just a position. A token that does not decode to exactly the expected position is rejected, so a
garbled or hand-edited one fails with ``400`` instead of silently restarting from the top.
"""

import base64
import binascii
import json
import re
from datetime import date

from settlement.api.schemas import BREAK_ID, MERCHANT_ID

MAX_CURSOR_LENGTH = 200


class InvalidCursorError(ValueError):
    """The cursor is malformed, tampered with, or belongs to another listing."""


def _encode(payload: dict[str, str]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _decode(token: str, keys: frozenset[str]) -> dict[str, str]:
    if not token or len(token) > MAX_CURSOR_LENGTH:
        raise InvalidCursorError("the cursor has an invalid length")
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise InvalidCursorError("the cursor is not valid base64url JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or not all(isinstance(v, str) for v in payload.values())
    ):
        raise InvalidCursorError("the cursor does not belong to this listing")
    return payload


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidCursorError("the cursor holds an invalid date") from exc


def _check(value: str, pattern: str, what: str) -> str:
    if not re.fullmatch(pattern, value):
        raise InvalidCursorError(f"the cursor holds an invalid {what}")
    return value


# --- settlements: ordered by (settlement_date, merchant_id) ------------------------------------

_SETTLEMENT_KEYS = frozenset({"d", "m"})


def encode_settlement_cursor(settlement_date: date, merchant_id: str) -> str:
    return _encode({"d": settlement_date.isoformat(), "m": merchant_id})


def decode_settlement_cursor(token: str) -> tuple[date, str]:
    payload = _decode(token, _SETTLEMENT_KEYS)
    return _parse_date(payload["d"]), _check(payload["m"], MERCHANT_ID, "merchant id")


# --- breaks: ordered by (settlement_date, break_id) --------------------------------------------

_BREAK_KEYS = frozenset({"d", "b"})


def encode_break_cursor(settlement_date: date, break_id: str) -> str:
    return _encode({"d": settlement_date.isoformat(), "b": break_id})


def decode_break_cursor(token: str) -> tuple[date, str]:
    payload = _decode(token, _BREAK_KEYS)
    return _parse_date(payload["d"]), _check(payload["b"], BREAK_ID, "break id")
