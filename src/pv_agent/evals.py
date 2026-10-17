"""Offline-reproducible evaluation harness for the agent.

Each question in evals/questions.jsonl declares behavioural expectations:
  expect_tools_any : at least one of these tools must be called ([] with expect_no_tools=true => none)
  contains_all     : every substring must appear in the answer (case-insensitive)
  contains_any     : at least one substring must appear
Metrics: tool-selection accuracy, content accuracy, numeric grounding rate
(share of numbers in the answer that appear in some tool output) and overall pass rate.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .agent import PharmacovigilanceAgent, AgentResponse

_NUM = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*")


def _floats(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _all_numbers(obj) -> list[float]:
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _all_numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _all_numbers(v)]
    if isinstance(obj, str):
        return _floats(obj)
    return []


def grounding_rate(resp: AgentResponse) -> float | None:
    """Fraction of 'material' numbers (not years/step lists) in the answer found in tool outputs."""
    answer_nums = [n for n in _floats(resp.answer) if not (1900 <= n <= 2100)]
    answer_nums = [n for n in answer_nums if n >= 1 or n != int(n)]
    if not answer_nums:
        return None
    pool = [n for c in resp.tool_calls if c.output for n in _all_numbers(c.output)]
    ok = sum(any(abs(n - p) <= max(0.06, 0.01 * abs(p)) for p in pool) for n in answer_nums)
    return ok / len(answer_nums)


def evaluate_case(case: dict, resp: AgentResponse) -> dict:
    called = {c.name for c in resp.tool_calls}
    answer = resp.answer.lower()
    if case.get("expect_no_tools"):
        tool_ok = not called
    elif case.get("expect_tools_any"):
        tool_ok = bool(called & set(case["expect_tools_any"]))
    else:
        tool_ok = True
    all_ok = all(s.lower() in answer for s in case.get("contains_all", []))
    any_list = case.get("contains_any", [])
    any_ok = (not any_list) or any(s.lower() in answer for s in any_list)
    content_ok = all_ok and any_ok
    g = grounding_rate(resp)
    return {"id": case["id"], "question": case["question"], "tools_called": sorted(called), "tool_ok": tool_ok,
            "content_ok": content_ok, "grounding": g, "passed": tool_ok and content_ok and (g is None or g >= 0.8),
            "steps": resp.steps, "tokens": resp.input_tokens + resp.output_tokens, "answer": resp.answer}


def run_eval(agent: PharmacovigilanceAgent, questions_path: str | Path, out_path: str | Path | None = None) -> dict:
    cases = [json.loads(l) for l in Path(questions_path).read_text().splitlines() if l.strip()]
    results = [evaluate_case(c, agent.ask(c["question"])) for c in cases]
    grounded = [r["grounding"] for r in results if r["grounding"] is not None]
    summary = {
        "n": len(results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "tool_accuracy": round(sum(r["tool_ok"] for r in results) / len(results), 3),
        "content_accuracy": round(sum(r["content_ok"] for r in results) / len(results), 3),
        "mean_grounding": round(statistics.mean(grounded), 3) if grounded else None,
        "mean_steps": round(statistics.mean(r["steps"] for r in results), 2),
        "mean_tokens": round(statistics.mean(r["tokens"] for r in results)),
    }
    report = {"summary": summary, "results": results}
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report
