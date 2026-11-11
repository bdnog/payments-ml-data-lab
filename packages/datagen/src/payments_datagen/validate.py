"""``validate``: check a generated directory against contracts, foreign keys and invariants."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from payments_datagen.generate import DATASET_PATHS
from payments_datagen.invariants import check_dataset_invariants, check_determinism
from payments_datagen.manifest import MANIFEST_NAME, read_manifest
from payments_datagen.profiles import TINY
from payments_datagen.storage import read_datasets
from payments_shared.data_contracts import Violation, load_contract, validate_dataframe


@dataclass
class Section:
    title: str
    summary: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


@dataclass
class Report:
    path: Path
    sections: list[Section] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(section.ok for section in self.sections)

    def render(self) -> str:
        lines = [f"validate {self.path}"]
        for section in self.sections:
            status = " OK " if section.ok else "FAIL"
            detail = section.summary if section.ok else f"{len(section.violations)} violation(s)"
            lines.append(f"  [{status}] {section.title:<16} {detail}")
            lines.extend(f"         - {violation}" for violation in section.violations)
        lines.append("RESULT: " + ("PASSED" if self.ok else "FAILED"))
        return "\n".join(lines)


def validate_directory(path: Path, *, determinism: bool = True) -> Report:
    report = Report(path)
    try:
        manifest = read_manifest(path / MANIFEST_NAME)
        datasets = read_datasets(path)
    except (FileNotFoundError, ValueError) as exc:
        failure = Violation("files", "unreadable", str(exc), 0)
        report.sections.append(Section("files", "", [failure]))
        return report
    report.sections.append(Section("files", f"{len(datasets)} datasets and a manifest"))

    contract_violations: list[Violation] = []
    for name, frame in datasets.items():
        contract_violations += validate_dataframe(frame, load_contract(name))
    report.sections.append(
        Section("data contracts", f"{len(datasets)} contracts honored", contract_violations)
    )

    counts = [
        Violation(
            "manifest", "row_count", f"{name}: manifest says {expected}, file has {frame.height}", 0
        )
        for name, frame in datasets.items()
        if (expected := manifest["row_counts"].get(name)) != frame.height
    ]
    report.sections.append(Section("manifest", "row counts match", counts))

    params = manifest["parameters"]
    last_day = date.fromisoformat(params["start_date"]) + timedelta(days=params["days"] - 1)
    report.sections.append(
        Section(
            "invariants",
            "keys, labels, ledger and PSP file agree",
            check_dataset_invariants(datasets, last_day),
        )
    )
    if determinism:
        report.sections.append(
            Section(
                "determinism",
                f"same seed equal, other seed different (profile {TINY.name})",
                check_determinism(TINY),
            )
        )
    assert set(DATASET_PATHS) == set(datasets)
    return report
