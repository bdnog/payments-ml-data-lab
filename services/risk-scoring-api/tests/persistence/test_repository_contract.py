"""The contract of ``DecisionRepository``, run against every adapter (memory and Postgres)."""

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime

from payments_shared.paths import contracts_dir
from risk_scoring.api.schemas import ScoreRequest
from risk_scoring.persistence.repository import (
    DecisionRecord,
    hash_request,
)

EXAMPLE = json.loads(
    (contracts_dir() / "examples" / "risk-scoring" / "score.request.json").read_text()
)


def record(transaction_id: str = "txn_0a1b2c3d4e5f", **patch) -> DecisionRecord:
    base = DecisionRecord(
        transaction_id=transaction_id,
        request_hash="a" * 64,
        request={"transaction_id": transaction_id, "amount_minor": 89_900, "nested": {"x": [1, 2]}},
        score=0.123456789012345,
        decision="REVIEW",
        model_version="mlp-test",
        reason_codes=["NEW_DEVICE", "PROXY_DETECTED"],
        latency_ms=1.5,
    )
    return replace(base, **patch)


def test_first_save_creates_and_returns_the_stored_row(storage):
    open_repository, count = storage
    with open_repository() as repository:
        result = repository.save_or_get(record())

    assert result.created
    stored = result.record
    assert stored.created_at is not None and stored.created_at.tzinfo is not None
    assert abs((datetime.now(UTC) - stored.created_at).total_seconds()) < 60
    assert replace(stored, created_at=None) == record()
    assert count() == 1


def test_values_survive_the_round_trip_exactly(storage):
    open_repository, _ = storage
    original = record(score=0.30000000000000004, latency_ms=0.1 + 0.2)
    with open_repository() as repository:
        repository.save_or_get(original)
    with open_repository() as repository:
        loaded = repository.get(original.transaction_id)

    assert loaded is not None
    assert loaded.score == original.score and loaded.latency_ms == original.latency_ms
    assert loaded.request == original.request and loaded.reason_codes == original.reason_codes


def test_saving_the_same_id_again_returns_the_first_row_untouched(storage):
    open_repository, count = storage
    with open_repository() as repository:
        first = repository.save_or_get(record(score=0.2, latency_ms=1.0))
    with open_repository() as repository:
        second = repository.save_or_get(record(score=0.9, latency_ms=7.0, decision="DECLINE"))

    assert first.created and not second.created
    assert second.record == first.record
    assert second.record.score == 0.2
    assert count() == 1


def test_a_different_payload_under_the_same_id_is_returned_for_the_caller_to_compare(storage):
    open_repository, count = storage
    with open_repository() as repository:
        repository.save_or_get(record(request_hash="a" * 64))
        result = repository.save_or_get(record(request_hash="b" * 64))

    assert not result.created
    assert result.record.request_hash == "a" * 64  # the stored hash: the API turns this into 409
    assert count() == 1


def test_get_returns_none_for_an_unknown_id(storage):
    open_repository, _ = storage
    with open_repository() as repository:
        assert repository.get("txn_ffffffffffff") is None


def test_different_ids_are_independent(storage):
    open_repository, count = storage
    with open_repository() as repository:
        a = repository.save_or_get(record("txn_00000000000a"))
        b = repository.save_or_get(record("txn_00000000000b"))
    assert a.created and b.created and count() == 2


def test_concurrent_inserts_of_one_id_leave_one_row_and_one_answer(storage):
    """Racing identical requests converge: one winner, and every caller sees the winner's row."""
    open_repository, count = storage
    n_threads = 12
    barrier = threading.Barrier(n_threads)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            with open_repository() as repository:
                barrier.wait(timeout=10)  # release all threads at once
                # Each racer computed a slightly different latency, like real concurrent requests.
                results[i] = repository.save_or_get(record(latency_ms=float(i), score=0.5))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert count() == 1
    assert sum(r.created for r in results) == 1
    winner = next(r.record for r in results if r.created)
    assert all(r.record.latency_ms == winner.latency_ms for r in results)
    assert all(r.record == winner for r in results)


# --- request hashing (pure) -------------------------------------------------------------------


def test_hash_is_stable_and_ignores_key_order():
    a = ScoreRequest.model_validate(EXAMPLE)
    reordered = dict(reversed(list(EXAMPLE.items())))
    b = ScoreRequest.model_validate(reordered)

    assert hash_request(a) == hash_request(b)
    assert len(hash_request(a)) == 64


def test_hash_changes_with_any_field():
    base = ScoreRequest.model_validate(EXAMPLE)
    changed = ScoreRequest.model_validate({**EXAMPLE, "amount_minor": EXAMPLE["amount_minor"] + 1})
    nested = ScoreRequest.model_validate(
        {**EXAMPLE, "device": {**EXAMPLE["device"], "is_proxy": not EXAMPLE["device"]["is_proxy"]}}
    )
    assert len({hash_request(base), hash_request(changed), hash_request(nested)}) == 3


def test_hash_of_the_example_is_pinned():
    """A change here means every stored hash stops matching: do it on purpose or not at all."""
    digest = hash_request(ScoreRequest.model_validate(EXAMPLE))
    assert digest == "4bb814b15ef04a84be80500c24a586ac659245ebaa246f6c2df423ef9a0e00fa"
