"""Locating the repository and its ``contracts/`` directory.

The contracts live in the repository, not inside the installed package, so code running from a
checkout finds them by walking up from this file. Anywhere else (for example a Docker image, where
contracts are not copied yet; see ADR 0004) set ``CONTRACTS_DIR``.
"""

import os
from pathlib import Path

CONTRACTS_DIR_ENV = "CONTRACTS_DIR"


def repo_root() -> Path:
    """Return the repository root: the nearest parent with both ``uv.lock`` and ``contracts/``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "uv.lock").is_file() and (parent / "contracts").is_dir():
            return parent
    raise FileNotFoundError(
        f"repository root not found above {__file__}; set ${CONTRACTS_DIR_ENV} to the contracts dir"
    )


def contracts_dir() -> Path:
    """Return the contracts directory, honoring ``$CONTRACTS_DIR`` when set."""
    override = os.environ.get(CONTRACTS_DIR_ENV)
    if override:
        return Path(override)
    return repo_root() / "contracts"
