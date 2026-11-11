"""The lifecycle of a run, as a pure state machine."""

from settlement.domain.enums import RunStatus

MAX_ERROR_LENGTH = 2_000

# PENDING -> RUNNING -> DONE | FAILED, and RUNNING -> PENDING when a stale run is given back.
ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({RunStatus.DONE, RunStatus.FAILED, RunStatus.PENDING}),
    RunStatus.DONE: frozenset(),
    RunStatus.FAILED: frozenset(),
}


class IllegalTransitionError(Exception):
    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(f"a run cannot go from {current} to {target}")
        self.current = current
        self.target = target


def check_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise IllegalTransitionError(current, target)


def truncate_error(message: str) -> str:
    """Errors are stored and served, so they are bounded."""
    return message if len(message) <= MAX_ERROR_LENGTH else message[: MAX_ERROR_LENGTH - 1] + "…"
