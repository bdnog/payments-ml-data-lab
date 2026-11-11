import subprocess
import sys

import pytest

from settlement.batch.__main__ import main


def test_run_with_valid_date_exits_zero(capsys):
    # configure_logging() replaces root handlers, so assert on stderr instead of caplog.
    assert main(["run", "--date", "2026-11-10"]) == 0
    stderr = capsys.readouterr().err
    assert "settlement run requested for 2026-11-10 (not implemented yet)" in stderr


@pytest.mark.parametrize("bad", ["2026-13-01", "10/11/2026", "20261110", "not-a-date"])
def test_run_with_invalid_date_exits_nonzero(bad):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--date", bad])

    assert exc.value.code != 0


def test_run_requires_date():
    with pytest.raises(SystemExit) as exc:
        main(["run"])

    assert exc.value.code != 0


def test_module_entry_point():
    ok = subprocess.run(
        [sys.executable, "-m", "settlement.batch", "run", "--date", "2026-11-10"],
        capture_output=True,
        text=True,
    )
    bad = subprocess.run(
        [sys.executable, "-m", "settlement.batch", "run", "--date", "nope"],
        capture_output=True,
        text=True,
    )

    assert ok.returncode == 0
    assert bad.returncode != 0
