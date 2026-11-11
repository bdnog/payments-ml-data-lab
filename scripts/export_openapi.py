"""Write the OpenAPI document of each service to ``contracts/openapi/<service>.v1.json``.

The documents are generated from the apps, never edited by hand. Output is deterministic (fixed API
version, sorted keys, stable formatting), so CI can regenerate them and fail on any difference:
the committed contract must always match the code.

Usage: ``uv run python scripts/export_openapi.py`` (or ``make contracts``).
"""

import json
from pathlib import Path

from payments_shared.paths import contracts_dir
from risk_scoring.api.main import create_app as create_risk_app
from risk_scoring.config import Settings as RiskSettings
from settlement.api.main import create_app as create_settlement_app
from settlement.config import Settings as SettlementSettings

# The version of the API contract (not of the build): bump it on a breaking change.
API_VERSION = "1.0.0"


def build_documents() -> dict[str, dict]:
    risk = create_risk_app(RiskSettings(app_version=API_VERSION, log_level="WARNING"))
    settlement = create_settlement_app(
        SettlementSettings(app_version=API_VERSION, log_level="WARNING")
    )
    return {
        "risk-scoring-api.v1.json": risk.openapi(),
        "settlement-service.v1.json": settlement.openapi(),
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    out_dir: Path = contracts_dir() / "openapi"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in build_documents().items():
        (out_dir / name).write_text(render(document), encoding="utf-8")
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
