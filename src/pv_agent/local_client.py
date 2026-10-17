"""Local rule-based / offline client that provides free tool-calling without an API key.

When ANTHROPIC_API_KEY is not set or set to 'mock'/'offline'/'local'/'free',
this client allows the entire agent, CLI, eval harness, and Streamlit UI to run
100% locally and free of charge.
"""
from __future__ import annotations

import difflib
import json
import re
from types import SimpleNamespace as NS
from typing import Any

from .tools import PharmacovigilanceTools


def _msg(blocks: list, stop_reason: str, in_tok: int = 40, out_tok: int = 25):
    return NS(content=blocks, stop_reason=stop_reason, usage=NS(input_tokens=in_tok, output_tokens=out_tok))


def _tool_use(id_: str, name: str, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def _text(t: str):
    return NS(type="text", text=t)


class LocalMessages:
    def __init__(self, tools: PharmacovigilanceTools | None = None):
        self.tools = tools

    def create(self, *, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> Any:
        last = messages[-1]
        last_content = last.get("content")

        # Step 2+: Tool result received from agent loop
        if isinstance(last_content, list) and last_content and last_content[0].get("type") == "tool_result":
            return self._handle_tool_result(messages)

        # Step 1: User question
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                user_query = m["content"]
                break

        return self._handle_user_query(user_query)

    def _handle_user_query(self, query: str):
        q = query.strip().lower()

        # 1. Medical advice check
        advice_patterns = [
            r"\bdose\b", r"\bdosage\b", r"\bhow much\b", r"\bshould i take\b",
            r"\bcan i take\b", r"\bprescribe\b", r"\bfor my\b", r"\btreat my\b",
            r"\bhow to take\b", r"\bmedication for\b"
        ]
        if any(re.search(pat, q) for pat in advice_patterns):
            text = (
                "I am an analytics assistant for adverse-event data, not a clinician or medical doctor. "
                "I cannot provide personal medical advice, dosage recommendations, or treatment decisions. "
                "Please consult a qualified healthcare professional, physician, doctor, or pharmacist."
            )
            return _msg([_text(text)], "end_turn")

        # 2. Destructive SQL / injection check
        destructive_sql = [
            r"\bdrop\s+table\b", r"\bdelete\s+from\b", r"\binsert\s+into\b",
            r"\bupdate\s+\w+\s+set\b", r"\balter\s+table\b", r"\btruncate\b"
        ]
        if any(re.search(pat, q) for pat in destructive_sql):
            text = (
                "I cannot run modifying or destructive SQL statements. Database access is strictly read-only "
                "and restricted to safe analytical queries."
            )
            return _msg([_text(text)], "end_turn")

        # 3. Database overview questions
        if any(phrase in q for phrase in [
            "how many cases", "overview", "what data", "period do they cover",
            "database overview", "summary of the database", "total cases", "dataset"
        ]) and not any(k in q for k in ["signals for", "adverse events for", "reactions for"]):
            return _msg([_tool_use("tu_overview", "get_database_overview")], "tool_use")

        # 4. Extract potential drugs from query
        drugs = self._extract_drugs(query)

        # 5. Drug comparison
        if len(drugs) >= 2 and any(k in q for k in ["compare", "differ", "difference", "versus", "vs", "between"]):
            return _msg([_tool_use("tu_compare", "compare_drugs", drug_a=drugs[0], drug_b=drugs[1])], "tool_use")

        primary_drug = drugs[0] if drugs else self._extract_unknown_drug_word(query)

        # 6. Safety signals / disproportionality
        if any(k in q for k in ["signal", "disproportionate", "disproportionality", "prr", "ror", "evans"]):
            if primary_drug:
                return _msg([_tool_use("tu_signal", "signal_detection", drug=primary_drug)], "tool_use")

        # 7. Reporting trends over time
        if any(k in q for k in ["trend", "over time", "over the years", "changed", "change", "history", "per year", "per quarter"]):
            if primary_drug:
                # check if a specific reaction is mentioned
                reaction = self._extract_reaction(query)
                inp = {"drug": primary_drug}
                if reaction:
                    inp["reaction"] = reaction
                return _msg([_tool_use("tu_trend", "reporting_trend", **inp)], "tool_use")

        # 8. Serious outcomes
        if any(k in q for k in ["outcome", "serious", "death", "fatal", "hospitalization", "life-threatening"]):
            if primary_drug:
                return _msg([_tool_use("tu_outcome", "outcome_breakdown", drug=primary_drug)], "tool_use")

        # 9. Demographics
        if any(k in q for k in ["age", "sex", "gender", "country", "demographic"]):
            if primary_drug:
                by = "sex" if any(k in q for k in ["sex", "gender"]) else "country" if "country" in q else "age_group"
                return _msg([_tool_use("tu_demo", "demographic_breakdown", drug=primary_drug, by=by)], "tool_use")

        # 10. General adverse events / top reactions
        if primary_drug:
            return _msg([_tool_use("tu_events", "top_adverse_events", drug=primary_drug)], "tool_use")

        # Default fallback: overview
        return _msg([_tool_use("tu_overview", "get_database_overview")], "tool_use")

    def _handle_tool_result(self, messages: list[dict]):
        # Find the last tool use and tool result
        tool_results = messages[-1]["content"]
        tr = tool_results[0]
        payload = json.loads(tr.get("content", "{}"))
        is_error = tr.get("is_error", False)

        # Find previous tool_use block
        tool_use = None
        for m in reversed(messages[:-1]):
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "tool_use" and b.get("id") == tr.get("tool_use_id"):
                        tool_use = b
                        break
            if tool_use:
                break

        tool_name = tool_use.get("name") if tool_use else ""
        tool_input = tool_use.get("input", {}) if tool_use else {}

        # Handle tool errors / typos
        if is_error or "error" in payload:
            err_msg = payload.get("error", "")
            # Check for typo correction hint: "Did you mean: PREDNISONE?"
            m = re.search(r"Did you mean:\s*([^?]+)\?", err_msg)
            if m:
                suggestions = [s.strip() for s in m.group(1).split(",")]
                if suggestions:
                    best_match = suggestions[0]
                    # Retry with the corrected drug name
                    new_input = dict(tool_input)
                    if "drug" in new_input:
                        new_input["drug"] = best_match
                    elif "drug_a" in new_input:
                        new_input["drug_a"] = best_match
                    return _msg([_tool_use(f"tu_retry_{best_match}", tool_name, **new_input)], "tool_use")

            # Unknown drug or no data found
            drug_name = tool_input.get("drug") or tool_input.get("query") or "the specified drug"
            answer = (
                f"No matching drug or reports for '{drug_name}' were found in the database. "
                "Please verify the medication name or active ingredient."
            )
            return _msg([_text(answer)], "end_turn")

        # Format grounded natural language answer from tool data
        answer = self._format_answer(tool_name, payload, messages)
        return _msg([_text(answer)], "end_turn")

    def _format_answer(self, tool_name: str, data: dict, messages: list[dict]) -> str:
        # Check original question context
        user_query = ""
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                user_query = m["content"]
                break

        if tool_name == "get_database_overview":
            source = data.get("data_source", "Database")
            n_cases = data.get("n_cases", 0)
            drugs = data.get("n_distinct_drugs", 0)
            reacs = data.get("n_distinct_reactions", 0)
            first = data.get("first_report", "")
            last = data.get("last_report", "")
            top_drugs = ", ".join(f"{d['drug']} ({d['cases']:,} cases)" for d in data.get("top_suspect_drugs", [])[:5])

            return (
                f"The database contains {n_cases:,} adverse event cases covering the reporting period from "
                f"{first} to {last}.\n\n"
                f"- **Data Source:** {source}\n"
                f"- **Distinct Drugs:** {drugs:,}\n"
                f"- **Distinct Reactions:** {reacs:,}\n"
                f"- **Top Suspect Drugs:** {top_drugs}\n\n"
                f"*(Note: Since this dataset is {source.lower()}, findings are illustrative and not real-world evidence.)*"
            )

        if tool_name == "signal_detection":
            drug = data.get("drug", "THE DRUG")
            suspect_cases = data.get("suspect_cases", 0)
            n_flagged = data.get("n_flagged", 0)
            signals = data.get("signals", [])
            flagged_signals = [s for s in signals if s.get("flagged")]

            lines = [
                f"Disproportionality analysis for **{drug}** across {suspect_cases:,} suspect cases "
                f"identified {n_flagged} safety signal(s) meeting Evans screening criteria (cases ≥ 3, PRR ≥ 2, chi² ≥ 4):\n"
            ]

            for s in flagged_signals[:5]:
                rx = s["reaction"]
                cases = s["cases"]
                prr = s["prr"]
                ror = s["ror"]
                chi2 = s["chi2"]
                lines.append(f"- **{rx}**: {cases} cases, PRR = {prr} (95% CI: {s['prr_ci_low']}–{s['prr_ci_high']}), ROR = {ror}, chi² = {chi2}")

            # If user asked about a specific reaction (e.g. lactic acidosis) that wasn't flagged or is among signals:
            q_lower = user_query.lower()
            for s in signals:
                if s["reaction"].lower() in q_lower and s not in flagged_signals[:5]:
                    lines.append(
                        f"\nRegarding **{s['reaction']}**: {s['cases']} cases reported, PRR = {s['prr']} "
                        f"(95% CI: {s['prr_ci_low']}–{s['prr_ci_high']}), ROR = {s['ror']}, chi² = {s['chi2']} "
                        f"({'flagged as a signal' if s.get('flagged') else 'did not meet signal criteria'})."
                    )

            lines.append(
                f"\n**Statistical Caveat:** {data.get('caveat', 'Disproportionality indicates statistical association in spontaneous reports, not causation or incidence.')}"
            )
            return "\n".join(lines)

        if tool_name == "compare_drugs":
            da, db = data.get("drug_a", "DRUG A"), data.get("drug_b", "DRUG B")
            ca, cb = data.get("cases_a", 0), data.get("cases_b", 0)
            diffs = data.get("largest_differences", [])

            lines = [
                f"Comparison of adverse reaction profiles between **{da}** ({ca:,} suspect cases) and **{db}** ({cb:,} suspect cases):\n"
            ]
            for d in diffs[:6]:
                rx = d["reaction"]
                pct_a = d.get(f"{da}_pct", 0)
                pct_b = d.get(f"{db}_pct", 0)
                diff = d.get("abs_diff_pct", 0)
                lines.append(f"- **{rx}**: {pct_a}% of {da} reports vs {pct_b}% of {db} reports (difference: {diff}%)")

            lines.append(
                "\n*Note: Percentages reflect the proportion of each drug's suspect cases reporting each reaction. Differences highlight divergent safety profiles in spontaneous reporting.*"
            )
            return "\n".join(lines)

        if tool_name == "reporting_trend":
            drug = data.get("drug", "DRUG")
            rx = data.get("reaction")
            series = data.get("series", [])
            gran = data.get("granularity", "year")

            rx_str = f" for **{rx}**" if rx else ""
            lines = [f"Reporting trend over time for **{drug}**{rx_str} by {gran}:\n"]
            for pt in series:
                period = pt["period"]
                cases = pt.get("cases_with_reaction") if rx else pt.get("cases", 0)
                lines.append(f"- **{period}**: {cases:,} reports")

            if len(series) >= 2:
                first_c = series[0].get("cases_with_reaction") if rx else series[0].get("cases", 0)
                last_c = series[-1].get("cases_with_reaction") if rx else series[-1].get("cases", 0)
                if last_c > first_c:
                    lines.append(f"\nOverall, reports have shown an **increasing trend**, rising from {first_c:,} reports in {series[0]['period']} to {last_c:,} reports in {series[-1]['period']}.")
                else:
                    lines.append(f"\nReports shifted from {first_c:,} reports in {series[0]['period']} to {last_c:,} reports in {series[-1]['period']}.")

            return "\n".join(lines)

        if tool_name == "outcome_breakdown":
            drug = data.get("drug", "DRUG")
            suspect = data.get("suspect_cases", 0)
            serious = data.get("cases_with_serious_outcome", 0)
            pct = data.get("pct_with_serious_outcome", 0)
            outcomes = data.get("outcomes", [])

            lines = [
                f"Outcomes breakdown for **{drug}** ({suspect:,} total suspect cases):\n",
                f"- **Serious outcomes reported:** {serious:,} cases ({pct}% of suspect reports)\n",
                f"Specific serious outcome breakdown:"
            ]
            for o in outcomes:
                lines.append(f"- **{o.get('outcome', o.get('code'))}**: {o.get('cases', 0):,} cases ({o.get('pct_of_cases', 0)}%)")

            return "\n".join(lines)

        if tool_name == "demographic_breakdown":
            drug = data.get("drug", "DRUG")
            suspect = data.get("suspect_cases", 0)
            by = data.get("by", "group")
            rows = data.get("breakdown", [])

            lines = [f"Demographic breakdown for **{drug}** ({suspect:,} suspect cases) by {by}:\n"]
            for r in rows:
                lines.append(f"- **{r.get('grp')}**: {r.get('cases', 0):,} cases ({r.get('pct_of_cases', 0)}%)")
            return "\n".join(lines)

        if tool_name == "top_adverse_events":
            drug = data.get("drug", "DRUG")
            suspect = data.get("suspect_cases", 0)
            reacs = data.get("top_reactions", [])

            lines = [f"Top reported adverse events for **{drug}** ({suspect:,} suspect cases):\n"]
            for r in reacs[:8]:
                lines.append(f"- **{r['reaction']}**: {r['cases']:,} cases ({r['pct_of_cases']}%)")
            return "\n".join(lines)

        return json.dumps(data, indent=2)

    def _extract_drugs(self, text: str) -> list[str]:
        words = re.findall(r"[A-Za-z\-]+", text.upper())
        known_drugs = [
            "ADALIMUMAB", "AMLODIPINE", "ATORVASTATIN", "DEXAMETHASONE", "HYDROCORTISONE",
            "IBUPROFEN", "LEVOTHYROXINE", "LISINOPRIL", "METFORMIN", "METHYLPREDNISOLONE",
            "OMEPRAZOLE", "PREDNISONE", "SEMAGLUTIDE", "SERTRALINE", "WARFARIN"
        ]
        found = []
        for w in words:
            if w in known_drugs and w not in found:
                found.append(w)

        # Check for near matches (typos like predisone)
        if not found:
            for w in words:
                if len(w) >= 5:
                    matches = difflib.get_close_matches(w, known_drugs, n=1, cutoff=0.7)
                    if matches and matches[0] not in found:
                        found.append(w.lower())  # Pass typo to test typo recovery in agent!

        return found

    def _extract_unknown_drug_word(self, text: str) -> str:
        # Check patterns like "for <word>" or "reported for <word>"
        m = re.search(r"\b(?:for|with|about)\s+([A-Za-z\-]+)", text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in {"my", "the", "a", "an", "all", "each", "pain", "back"}:
                return candidate
        return ""

    def _extract_reaction(self, text: str) -> str | None:
        known_reactions = [
            "NAUSEA", "FATIGUE", "HEADACHE", "VOMITING", "PAIN", "DIARRHOEA", "DIZZINESS",
            "RASH", "DYSPNOEA", "PYREXIA", "HYPERGLYCAEMIA", "PRURITUS", "ASTHENIA",
            "ARTHRALGIA", "ANXIETY", "LACTIC ACIDOSIS", "ADRENAL INSUFFICIENCY",
            "OSTEOPOROSIS", "PANCREATITIS", "CUSHINGOID", "GASTROINTESTINAL HAEMORRHAGE",
            "MYALGIA", "RHABDOMYOLYSIS", "RENAL FAILURE ACUTE"
        ]
        t = text.upper()
        for rx in known_reactions:
            if rx in t:
                return rx
        return None


class LocalClient:
    """Drop-in Anthropic client replacement for offline / free local operation."""
    def __init__(self, tools: PharmacovigilanceTools | None = None):
        self.messages = LocalMessages(tools)
