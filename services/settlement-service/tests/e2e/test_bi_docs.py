"""Every SQL query in docs/bi-access.md runs against the marts of a real run."""

import re
from pathlib import Path

import duckdb
import pytest

from payments_shared.paths import repo_root
from settlement.batch.__main__ import main

DOC = repo_root() / "docs" / "bi-access.md"
QUERIES = re.findall(r"```sql\n(.*?)```", DOC.read_text(), flags=re.DOTALL)


@pytest.fixture(scope="module")
def marts(tmp_path_factory, sample_dir) -> Path:
    directory = tmp_path_factory.mktemp("bi-marts")
    assert (
        main(["run-range", "--all", "--input-dir", str(sample_dir), "--marts-dir", str(directory)])
        == 0
    )
    return directory


def test_the_page_documents_several_queries():
    assert len(QUERIES) >= 7


@pytest.mark.parametrize("query", QUERIES, ids=[q.strip().split("\n")[0][:60] for q in QUERIES])
def test_every_documented_query_runs_and_returns_rows(marts, query):
    rows = duckdb.sql(query.replace("<MARTS>", str(marts))).fetchall()
    assert rows, "the documented query returned nothing on the sample"
