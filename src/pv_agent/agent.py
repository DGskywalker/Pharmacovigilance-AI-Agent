"""LLM agent: an explicit tool-calling loop over the Anthropic Messages API.

Design notes
- The model never sees raw SQL access by default; it picks among typed analysis tools.
- All numbers must come from tool results (enforced by the system prompt and checked in evals).
- Tool failures are returned to the model as errors so it can self-correct (e.g. fix a drug name).
- The loop is bounded (`max_steps`) and every call is traced for transparency in the UI.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .tools import TOOL_SCHEMAS, PharmacovigilanceTools, ToolError, to_json

SYSTEM_PROMPT = """You are a pharmacovigilance analyst assistant working over a database of spontaneous adverse-event reports (FDA FAERS format).

Rules:
1. Use the provided tools to answer. Never state a count, percentage, ratio or date that did not come from a tool result. If you cannot obtain a number with the tools, say so.
2. If a tool returns an error (for example an unknown drug name), correct the call (try search_drugs) or explain the problem; do not guess.
3. Call get_database_overview when you need to know whether the data is real or synthetic. If it is synthetic, say clearly that findings are illustrative and not real-world evidence.
4. Always interpret disproportionality results correctly: PRR/ROR/chi-square show a statistical association in spontaneous reports. They do NOT prove causation and cannot estimate incidence, because spontaneous reporting has no denominator, under-reporting, stimulated reporting and duplicate/confounding issues. State this briefly whenever you present signals.
5. You are an analytics assistant, not a clinician. Do not give personal medical advice, dosing, or treatment recommendations; politely redirect such requests to a healthcare professional.
6. Be concise and structured: lead with the answer, then the key supporting numbers (cite the tool metric names), then caveats. Prefer one or two well-chosen tool calls over many.
"""


@dataclass
class ToolCall:
    name: str
    input: dict
    output: dict | None
    is_error: bool
    duration_ms: int


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    truncated: bool = False


def _block_to_dict(block) -> dict:
    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    raise ValueError(f"Unsupported content block type: {t}")


class PharmacovigilanceAgent:
    def __init__(self, settings: Settings | None = None, client=None, tools: PharmacovigilanceTools | None = None):
        self.settings = settings or get_settings()
        self.tools = tools or PharmacovigilanceTools(
            self.settings.db_path, self.settings.sql_row_limit, self.settings.sql_timeout_s)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not key or key.lower() in {"your-key-here", "none", "mock", "offline", "local", "free"}:
                from .local_client import LocalClient

                self._client = LocalClient(self.tools)
            else:
                import anthropic

                self._client = anthropic.Anthropic()
        return self._client

    def ask(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        """Answer a question. `history` is a list of prior {"role","content"(str)} turns."""
        messages = [*(history or []), {"role": "user", "content": question}]
        resp = AgentResponse(answer="", model=self.settings.model)

        for step in range(1, self.settings.max_steps + 1):
            resp.steps = step
            msg = self.client.messages.create(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            usage = getattr(msg, "usage", None)
            resp.input_tokens += getattr(usage, "input_tokens", 0) or 0
            resp.output_tokens += getattr(usage, "output_tokens", 0) or 0
            content = [_block_to_dict(b) for b in msg.content]
            messages.append({"role": "assistant", "content": content})

            tool_uses = [b for b in content if b["type"] == "tool_use"]
            if msg.stop_reason != "tool_use" or not tool_uses:
                resp.answer = "\n".join(b["text"] for b in content if b["type"] == "text").strip()
                return resp

            results = []
            for tu in tool_uses:
                t0 = time.perf_counter()
                try:
                    out = self.tools.call(tu["name"], tu["input"])
                    payload, is_err = out.data, False
                    if out.chart:
                        resp.charts.append(out.chart)
                except ToolError as e:
                    payload, is_err = {"error": str(e)}, True
                except Exception as e:  # never let a tool crash the loop
                    payload, is_err = {"error": f"Internal tool error: {type(e).__name__}: {e}"}, True
                resp.tool_calls.append(ToolCall(tu["name"], tu["input"], payload, is_err,
                                                int((time.perf_counter() - t0) * 1000)))
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": to_json(payload, self.settings.tool_result_char_limit),
                                **({"is_error": True} if is_err else {})})
            messages.append({"role": "user", "content": results})

        resp.truncated = True
        resp.answer = ("I reached the step limit before finishing. Here is what I gathered so far - "
                       "please narrow the question or ask a follow-up.")
        return resp


def trace_to_text(resp: AgentResponse) -> str:
    lines = []
    for i, c in enumerate(resp.tool_calls, 1):
        flag = " [ERROR]" if c.is_error else ""
        lines.append(f"{i}. {c.name}({json.dumps(c.input)}) - {c.duration_ms} ms{flag}")
    return "\n".join(lines)
