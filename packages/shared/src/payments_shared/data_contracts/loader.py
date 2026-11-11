"""Loading data contracts from ``contracts/data/*.yaml``."""

from pathlib import Path

import yaml

from payments_shared.data_contracts.models import Contract
from payments_shared.paths import contracts_dir


def data_contracts_dir() -> Path:
    return contracts_dir() / "data"


def list_contracts() -> list[str]:
    """Names of all contracts available, sorted."""
    return sorted(p.stem for p in data_contracts_dir().glob("*.yaml"))


def load_contract(name: str) -> Contract:
    """Load and structurally validate ``contracts/data/<name>.yaml``."""
    path = data_contracts_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"data contract {name!r} not found at {path}")
    contract = Contract.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if contract.name != name:
        raise ValueError(f"{path.name}: contract name {contract.name!r} does not match file name")
    return contract
