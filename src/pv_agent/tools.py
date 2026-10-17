"""Analysis tools exposed to the LLM agent.

Every number the agent reports should come from one of these functions, which
query the database deterministically. Each tool returns JSON-serializable data
and (optionally) a chart spec that the UI can render.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import OUTCOME_LABELS, connect_ro, run_with_limits
from .guardrails import SQLRejected, validate_sql
from .stats import disproportionality


class ToolError(Exception):
    """A user-correctable problem (bad drug name, no data, ...). Returned to the model, not raised to the user."""


@dataclass
class ToolResult:
    data: dict
    chart: dict | None = None


def _chart(kind: str, title: str, x: list, series: dict[str, list], x_label: str = "", y_label: str = "") -> dict:
    return {"type": kind, "title": title, "x": x, "series": series, "x_label": x_label, "y_label": y_label}


class PharmacovigilanceTools:
    def __init__(self, db_path: str | Path, row_limit: int = 200, timeout_s: float = 5.0):
        self.conn = connect_ro(db_path)
        self.row_limit = row_limit
        self.timeout_s = timeout_s
        self.n_cases = self._scalar("SELECT COUNT(*) FROM demo")

    # ---- helpers ---------------------------------------------------------
    def _q(self, sql: str, params=()) -> list[dict]:
        return run_with_limits(self.conn, sql, params, self.timeout_s)

    def _scalar(self, sql: str, params=()) -> Any:
        rows = self._q(sql, params)
        return next(iter(rows[0].values())) if rows else None

    def _resolve(self, drug: str) -> list[str]:
        """Map a user-supplied drug name to the set of drugname strings in the DB."""
        if not drug or not str(drug).strip():
            raise ToolError("A drug name is required.")
        n = str(drug).strip().upper()
        names = [r["drugname"] for r in self._q(
            "SELECT DISTINCT drugname FROM drug WHERE drugname = ? OR prod_ai = ? LIMIT 200", (n, n))]
        if not names:
            names = [r["drugname"] for r in self._q(
                "SELECT DISTINCT drugname FROM drug WHERE drugname LIKE ? OR prod_ai LIKE ? LIMIT 200",
                (f"%{n}%", f"%{n}%"))]
        if not names:
            pool = [r["drugname"] for r in self._q(
                "SELECT drugname FROM drug GROUP BY drugname ORDER BY COUNT(*) DESC LIMIT 5000")]
            close = difflib.get_close_matches(n, pool, n=5, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ToolError(f"No drug matching '{drug}' was found in the database.{hint}")
        return names

    def _cte(self, names: list[str], suspect_only: bool) -> tuple[str, list]:
        ph = ",".join("?" * len(names))
        role = " AND d.role_cod IN ('PS','SS')" if suspect_only else ""
        return f"dc AS (SELECT DISTINCT d.primaryid FROM drug d WHERE d.drugname IN ({ph}){role})", list(names)

    def _drug_cases(self, drug: str, suspect_only: bool = True):
        names = self._resolve(drug)
        cte, params = self._cte(names, suspect_only)
        n = self._scalar(f"WITH {cte} SELECT COUNT(*) FROM dc", params)
        if not n:
            raise ToolError(f"'{drug}' exists in the database but has no {'suspect-drug ' if suspect_only else ''}reports.")
        return names, cte, params, n

    # ---- tools -----------------------------------------------------------
    def get_database_overview(self) -> ToolResult:
        meta = {r["key"]: r["value"] for r in self._q("SELECT key, value FROM meta")}
        top = self._q("SELECT prod_ai AS drug, COUNT(DISTINCT primaryid) AS cases FROM drug "
                      "WHERE role_cod IN ('PS','SS') AND prod_ai <> '' GROUP BY prod_ai ORDER BY cases DESC LIMIT 10")
        span = self._q("SELECT MIN(report_date) AS first_report, MAX(report_date) AS last_report FROM demo")[0]
        return ToolResult({
            "data_source": meta.get("source"), "built_at": meta.get("built_at"),
            "n_cases": self.n_cases,
            "n_distinct_drugs": self._scalar("SELECT COUNT(DISTINCT prod_ai) FROM drug"),
            "n_distinct_reactions": self._scalar("SELECT COUNT(*) FROM reac_counts"),
            **span, "top_suspect_drugs": top,
        })

    def search_drugs(self, query: str, limit: int = 10) -> ToolResult:
        q = f"%{str(query).strip().upper()}%"
        rows = self._q("SELECT drugname, prod_ai AS active_ingredient, COUNT(DISTINCT primaryid) AS cases FROM drug "
                       "WHERE drugname LIKE ? OR prod_ai LIKE ? GROUP BY drugname, prod_ai ORDER BY cases DESC LIMIT ?",
                       (q, q, min(int(limit), 50)))
        return ToolResult({"query": query, "matches": rows})

    def top_adverse_events(self, drug: str, limit: int = 10, serious_only: bool = False) -> ToolResult:
        names, cte, params, n = self._drug_cases(drug)
        extra = " JOIN (SELECT DISTINCT primaryid FROM outc) o ON o.primaryid = r.primaryid" if serious_only else ""
        rows = self._q(f"WITH {cte} SELECT r.pt AS reaction, COUNT(DISTINCT r.primaryid) AS cases FROM reac r "
                       f"JOIN dc ON dc.primaryid = r.primaryid{extra} GROUP BY r.pt ORDER BY cases DESC LIMIT ?",
                       [*params, min(int(limit), 50)])
        denom = n
        if serious_only:
            denom = self._scalar(f"WITH {cte} SELECT COUNT(*) FROM dc JOIN (SELECT DISTINCT primaryid FROM outc) o "
                                 f"ON o.primaryid = dc.primaryid", params) or 1
        for r in rows:
            r["pct_of_cases"] = round(100 * r["cases"] / denom, 1)
        return ToolResult(
            {"drug": drug.upper(), "suspect_cases": n, "serious_only": serious_only,
             "denominator_cases": denom, "top_reactions": rows},
            _chart("bar", f"Top reported reactions: {drug.upper()}", [r["reaction"] for r in rows],
                   {"Cases": [r["cases"] for r in rows]}, "Reaction", "Cases"))

    def signal_detection(self, drug: str, limit: int = 10, min_cases: int = 3) -> ToolResult:
        names, cte, params, n = self._drug_cases(drug)
        rows = self._q(
            f"WITH {cte} SELECT r.pt AS pt, COUNT(DISTINCT r.primaryid) AS a, rc.n_cases AS total FROM reac r "
            f"JOIN dc ON dc.primaryid = r.primaryid JOIN reac_counts rc ON rc.pt = r.pt GROUP BY r.pt HAVING a >= ?",
            [*params, max(int(min_cases), 1)])
        out = []
        for r in rows:
            a = r["a"]; b = n - a; c = r["total"] - a; d = self.n_cases - n - c
            if min(b, c, d) < 0:  # reaction/drug overlap edge case
                continue
            out.append({"reaction": r["pt"], **disproportionality(a, b, c, d)})
        out.sort(key=lambda x: (x["flagged"], x["ror_ci_low"]), reverse=True)
        top = out[: min(int(limit), 30)]
        return ToolResult(
            {"drug": drug.upper(), "suspect_cases": n, "total_cases": self.n_cases,
             "method": "PRR and ROR with 95% CI; flagged = Evans criteria (cases>=3, PRR>=2, chi2>=4)",
             "n_reactions_tested": len(out), "n_flagged": sum(x["flagged"] for x in out), "signals": top,
             "caveat": "Disproportionality indicates a statistical association in spontaneous reports, not causation."},
            _chart("bar", f"Disproportionality (PRR): {drug.upper()}", [x["reaction"] for x in top],
                   {"PRR": [x["prr"] for x in top]}, "Reaction", "PRR"))

    def outcome_breakdown(self, drug: str) -> ToolResult:
        names, cte, params, n = self._drug_cases(drug)
        rows = self._q(f"WITH {cte} SELECT o.outc_cod AS code, COUNT(DISTINCT o.primaryid) AS cases FROM outc o "
                       f"JOIN dc ON dc.primaryid = o.primaryid GROUP BY o.outc_cod ORDER BY cases DESC", params)
        with_outcome = self._scalar(f"WITH {cte} SELECT COUNT(DISTINCT o.primaryid) FROM outc o JOIN dc ON dc.primaryid = o.primaryid", params)
        for r in rows:
            r["outcome"] = OUTCOME_LABELS.get(r["code"], r["code"])
            r["pct_of_cases"] = round(100 * r["cases"] / n, 1)
        return ToolResult(
            {"drug": drug.upper(), "suspect_cases": n, "cases_with_serious_outcome": with_outcome,
             "pct_with_serious_outcome": round(100 * with_outcome / n, 1), "outcomes": rows},
            _chart("bar", f"Serious outcomes: {drug.upper()}", [r["outcome"] for r in rows],
                   {"Cases": [r["cases"] for r in rows]}, "Outcome", "Cases"))

    def demographic_breakdown(self, drug: str, by: str = "age_group") -> ToolResult:
        if by not in ("age_group", "sex", "country"):
            raise ToolError("'by' must be one of: age_group, sex, country.")
        names, cte, params, n = self._drug_cases(drug)
        rows = self._q(f"WITH {cte} SELECT COALESCE(m.{by}, 'Unknown') AS grp, COUNT(*) AS cases FROM demo m "
                       f"JOIN dc ON dc.primaryid = m.primaryid GROUP BY grp ORDER BY cases DESC LIMIT 15", params)
        for r in rows:
            r["pct_of_cases"] = round(100 * r["cases"] / n, 1)
        return ToolResult({"drug": drug.upper(), "suspect_cases": n, "by": by, "breakdown": rows},
                          _chart("bar", f"{drug.upper()} cases by {by}", [r["grp"] for r in rows],
                                 {"Cases": [r["cases"] for r in rows]}, by, "Cases"))

    def compare_drugs(self, drug_a: str, drug_b: str, limit: int = 8) -> ToolResult:
        res = {}
        for label, drug in (("a", drug_a), ("b", drug_b)):
            names, cte, params, n = self._drug_cases(drug)
            rows = self._q(f"WITH {cte} SELECT r.pt AS reaction, COUNT(DISTINCT r.primaryid) AS cases FROM reac r "
                           f"JOIN dc ON dc.primaryid = r.primaryid GROUP BY r.pt", params)
            res[label] = (n, {r["reaction"]: r["cases"] for r in rows})
        (na, ca), (nb, cb) = res["a"], res["b"]
        rows = []
        for rx in set(ca) | set(cb):
            ra, rb = 100 * ca.get(rx, 0) / na, 100 * cb.get(rx, 0) / nb
            rows.append({"reaction": rx, f"{drug_a.upper()}_pct": round(ra, 1), f"{drug_b.upper()}_pct": round(rb, 1),
                         "abs_diff_pct": round(abs(ra - rb), 1)})
        rows.sort(key=lambda r: r["abs_diff_pct"], reverse=True)
        rows = rows[: min(int(limit), 20)]
        A, B = f"{drug_a.upper()}_pct", f"{drug_b.upper()}_pct"
        return ToolResult(
            {"drug_a": drug_a.upper(), "cases_a": na, "drug_b": drug_b.upper(), "cases_b": nb,
             "note": "Percentages are the share of each drug's suspect cases reporting the reaction; "
                     "rows are ranked by absolute difference.", "largest_differences": rows},
            _chart("grouped_bar", f"Reaction rates: {drug_a.upper()} vs {drug_b.upper()}", [r["reaction"] for r in rows],
                   {drug_a.upper(): [r[A] for r in rows], drug_b.upper(): [r[B] for r in rows]}, "Reaction", "% of cases"))

    def reporting_trend(self, drug: str, reaction: str | None = None, granularity: str = "year") -> ToolResult:
        if granularity not in ("year", "quarter"):
            raise ToolError("granularity must be 'year' or 'quarter'.")
        col = "report_year" if granularity == "year" else "report_quarter"
        names, cte, params, n = self._drug_cases(drug)
        if reaction:
            rx = str(reaction).strip().upper()
            rows = self._q(f"WITH {cte} SELECT m.{col} AS period, COUNT(DISTINCT m.primaryid) AS cases_with_reaction FROM demo m "
                           f"JOIN dc ON dc.primaryid = m.primaryid JOIN reac r ON r.primaryid = m.primaryid AND r.pt = ? "
                           f"WHERE m.{col} IS NOT NULL GROUP BY period ORDER BY period", [*params, rx])
            key, title = "cases_with_reaction", f"{drug.upper()} - {rx}: reports per {granularity}"
        else:
            rows = self._q(f"WITH {cte} SELECT m.{col} AS period, COUNT(DISTINCT m.primaryid) AS cases FROM demo m "
                           f"JOIN dc ON dc.primaryid = m.primaryid WHERE m.{col} IS NOT NULL GROUP BY period ORDER BY period", params)
            key, title = "cases", f"{drug.upper()}: reports per {granularity}"
        if not rows:
            raise ToolError("No reports found for that drug/reaction combination.")
        return ToolResult({"drug": drug.upper(), "reaction": reaction.upper() if reaction else None,
                           "granularity": granularity, "series": rows},
                          _chart("line", title, [str(r["period"]) for r in rows], {"Cases": [r[key] for r in rows]},
                                 granularity.title(), "Cases"))

    def run_sql(self, query: str) -> ToolResult:
        """Escape hatch for questions the structured tools can't answer. Heavily guarded."""
        try:
            safe = validate_sql(query, self.row_limit)
        except SQLRejected as e:
            raise ToolError(str(e)) from e
        try:
            rows = run_with_limits(self.conn, safe, (), self.timeout_s, authorize=True)
        except Exception as e:  # sqlite3.OperationalError, DatabaseError, interrupted
            raise ToolError(f"SQL error: {e}") from e
        return ToolResult({"row_count": len(rows), "row_limit": self.row_limit, "rows": rows,
                           "schema_hint": "tables: demo, drug, reac, outc, reac_counts, meta"})

    # ---- dispatch --------------------------------------------------------
    def call(self, name: str, args: dict) -> ToolResult:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            raise ToolError(f"Unknown tool '{name}'.")
        try:
            return fn(**(args or {}))
        except TypeError as e:
            raise ToolError(f"Invalid arguments for {name}: {e}") from e


_DRUG = {"type": "string", "description": "Generic or brand drug name, e.g. 'prednisone'."}

TOOL_SCHEMAS: list[dict] = [
    {"name": "get_database_overview",
     "description": "Describe the loaded dataset: source (real FAERS or synthetic), case counts, date range, top suspect drugs. "
                    "Call this first for general questions about the data, and to learn whether data is synthetic.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_drugs",
     "description": "Find drug names / active ingredients in the database matching a text fragment, with case counts. "
                    "Use when a drug name is ambiguous or a lookup failed.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
                      "required": ["query"]}},
    {"name": "top_adverse_events",
     "description": "Most frequently reported adverse reactions (MedDRA preferred terms) for a drug as suspect, with counts and % of cases.",
     "input_schema": {"type": "object", "properties": {"drug": _DRUG, "limit": {"type": "integer", "default": 10},
                                                       "serious_only": {"type": "boolean", "default": False}}, "required": ["drug"]}},
    {"name": "signal_detection",
     "description": "Disproportionality analysis (PRR, ROR with 95% CI, chi-square) of every reaction for a drug versus all other drugs. "
                    "Use for 'safety signals', 'unexpected reactions', 'what is disproportionately reported'.",
     "input_schema": {"type": "object", "properties": {"drug": _DRUG, "limit": {"type": "integer", "default": 10},
                                                       "min_cases": {"type": "integer", "default": 3}}, "required": ["drug"]}},
    {"name": "outcome_breakdown",
     "description": "Distribution of serious outcomes (death, hospitalization, life-threatening, ...) among a drug's suspect cases.",
     "input_schema": {"type": "object", "properties": {"drug": _DRUG}, "required": ["drug"]}},
    {"name": "demographic_breakdown",
     "description": "Breakdown of a drug's suspect cases by age_group, sex or country.",
     "input_schema": {"type": "object", "properties": {"drug": _DRUG, "by": {"type": "string", "enum": ["age_group", "sex", "country"]}},
                      "required": ["drug"]}},
    {"name": "compare_drugs",
     "description": "Compare the reaction profiles of two drugs; returns the reactions whose reporting rates differ most.",
     "input_schema": {"type": "object", "properties": {"drug_a": _DRUG, "drug_b": _DRUG, "limit": {"type": "integer", "default": 8}},
                      "required": ["drug_a", "drug_b"]}},
    {"name": "reporting_trend",
     "description": "Number of reports over time for a drug, optionally restricted to one reaction (MedDRA preferred term, e.g. 'PANCREATITIS').",
     "input_schema": {"type": "object", "properties": {"drug": _DRUG, "reaction": {"type": "string"},
                                                       "granularity": {"type": "string", "enum": ["year", "quarter"], "default": "year"}},
                      "required": ["drug"]}},
    {"name": "run_sql",
     "description": "LAST RESORT: run one read-only SQLite SELECT over tables demo(primaryid, age_years, age_group, sex, weight_kg, country, "
                    "report_date, report_year, report_quarter), drug(primaryid, drug_seq, role_cod, drugname, prod_ai), reac(primaryid, pt), "
                    "outc(primaryid, outc_cod), reac_counts(pt, n_cases). role_cod: PS=primary suspect, SS=secondary suspect, C=concomitant. "
                    "Prefer the structured tools whenever possible.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}


def to_json(data: dict, limit: int) -> str:
    s = json.dumps(data, default=str)
    if len(s) > limit:
        s = s[:limit] + '..."[truncated]"'
    return s
