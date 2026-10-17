import pytest

from pv_agent.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolError


def test_schemas_match_methods(tools):
    assert TOOL_NAMES == {t["name"] for t in TOOL_SCHEMAS}
    for name in TOOL_NAMES:
        assert callable(getattr(tools, name))


def test_overview_flags_synthetic(tools):
    d = tools.get_database_overview().data
    assert "SYNTHETIC" in d["data_source"] and d["n_cases"] == 6000


def test_embedded_signals_recovered(tools):
    """The pipeline must rediscover the associations planted in the synthetic data."""
    for drug, expected in [("prednisone", "OSTEOPOROSIS"), ("metformin", "LACTIC ACIDOSIS"),
                           ("ibuprofen", "GASTROINTESTINAL HAEMORRHAGE"), ("warfarin", "HAEMORRHAGE")]:
        res = tools.signal_detection(drug, limit=10).data
        flagged = {s["reaction"] for s in res["signals"] if s["flagged"]}
        assert expected in flagged, f"{drug}: {expected} not flagged (got {flagged})"


def test_background_reaction_not_flagged(tools):
    res = tools.signal_detection("prednisone", limit=30).data
    by = {s["reaction"]: s for s in res["signals"]}
    assert "HEADACHE" not in by or not by["HEADACHE"]["flagged"]


def test_brand_name_resolves(tools):
    assert tools.top_adverse_events("HUMIRA").data["suspect_cases"] > 0


def test_unknown_drug_suggests(tools):
    with pytest.raises(ToolError, match="Did you mean"):
        tools.top_adverse_events("predisone")


def test_top_events_and_pct(tools):
    d = tools.top_adverse_events("metformin", limit=5).data
    assert len(d["top_reactions"]) == 5
    assert all(0 < r["pct_of_cases"] <= 100 for r in d["top_reactions"])


def test_semaglutide_trend_is_increasing(tools):
    s = tools.reporting_trend("semaglutide").data["series"]
    assert s[-1]["cases"] > s[0]["cases"]


def test_compare_and_outcomes_and_demo(tools):
    assert tools.compare_drugs("ibuprofen", "atorvastatin").chart["type"] == "grouped_bar"
    assert tools.outcome_breakdown("warfarin").data["pct_with_serious_outcome"] > 0
    assert tools.demographic_breakdown("prednisone", by="sex").data["breakdown"]
    with pytest.raises(ToolError):
        tools.demographic_breakdown("prednisone", by="shoe_size")


def test_run_sql_ok_and_blocked(tools):
    assert tools.run_sql("SELECT COUNT(*) AS n FROM demo").data["rows"][0]["n"] == 6000
    with pytest.raises(ToolError):
        tools.run_sql("DROP TABLE demo")


def test_database_is_read_only(db_path):
    import sqlite3

    from pv_agent.db import connect_ro

    conn = connect_ro(db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM demo")


def test_authorizer_blocks_writes_even_if_guardrail_bypassed(tools):
    from pv_agent.db import run_with_limits

    with pytest.raises(Exception):
        run_with_limits(tools.conn, "CREATE TEMP TABLE x(a)", authorize=True)


def test_unknown_tool_and_bad_args(tools):
    with pytest.raises(ToolError):
        tools.call("nope", {})
    with pytest.raises(ToolError):
        tools.call("top_adverse_events", {"wrong": 1})
