import pytest

from pv_agent.guardrails import SQLRejected, validate_sql


@pytest.mark.parametrize("bad", [
    "DROP TABLE demo", "DELETE FROM drug", "SELECT 1; DROP TABLE demo", "UPDATE demo SET sex='M'",
    "PRAGMA table_info(demo)", "ATTACH DATABASE 'x.db' AS x", "SELECT 1 -- comment", "", "   ",
    "INSERT INTO meta VALUES ('a','b')", "WITH x AS (SELECT 1) DELETE FROM demo",
])
def test_rejects_unsafe(bad):
    with pytest.raises(SQLRejected):
        validate_sql(bad)


def test_accepts_select_and_wraps_limit():
    out = validate_sql("SELECT pt FROM reac;", row_limit=50)
    assert out.endswith("LIMIT 50") and out.startswith("SELECT * FROM (")


def test_accepts_cte():
    assert "WITH" in validate_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
