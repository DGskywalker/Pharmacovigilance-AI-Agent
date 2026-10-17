"""Agent loop tests using a scripted fake Anthropic client (no network / API key)."""
from types import SimpleNamespace as NS

from pv_agent.agent import PharmacovigilanceAgent
from pv_agent.config import Settings
from pv_agent.evals import evaluate_case, grounding_rate


def _msg(blocks, stop):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


def _tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def _text(t):
    return NS(type="text", text=t)


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def _agent(db_path, script, max_steps=5):
    s = Settings(db_path=db_path, model="fake", max_steps=max_steps, max_tokens=512)
    return PharmacovigilanceAgent(s, client=FakeClient(script))


def test_tool_roundtrip(db_path):
    agent = _agent(db_path, [
        _msg([_tool_use("t1", "signal_detection", drug="prednisone", limit=3)], "tool_use"),
        _msg([_text("Top signal: OSTEOPOROSIS.")], "end_turn"),
    ])
    r = agent.ask("signals for prednisone?")
    assert r.answer.startswith("Top signal") and r.steps == 2
    assert r.tool_calls[0].name == "signal_detection" and not r.tool_calls[0].is_error
    assert r.charts and r.input_tokens == 20
    # the tool_result was fed back to the model on the 2nd call
    second = agent.client.calls[1]["messages"][-2]["content"][0]
    assert second["type"] == "tool_result" and second["tool_use_id"] == "t1"


def test_tool_error_is_returned_to_model_not_raised(db_path):
    agent = _agent(db_path, [
        _msg([_tool_use("t1", "top_adverse_events", drug="foobarinol")], "tool_use"),
        _msg([_text("That drug was not found.")], "end_turn"),
    ])
    r = agent.ask("events for foobarinol?")
    assert r.tool_calls[0].is_error
    assert agent.client.calls[1]["messages"][-2]["content"][0]["is_error"] is True


def test_step_limit(db_path):
    loop = [_msg([_tool_use(f"t{i}", "get_database_overview")], "tool_use") for i in range(10)]
    r = _agent(db_path, loop, max_steps=3).ask("loop forever")
    assert r.truncated and r.steps == 3


def test_eval_scoring_and_grounding(db_path):
    agent = _agent(db_path, [
        _msg([_tool_use("t1", "get_database_overview")], "tool_use"),
        _msg([_text("The database holds 6000 cases (synthetic).")], "end_turn"),
    ])
    r = agent.ask("how many cases?")
    assert grounding_rate(r) == 1.0
    case = {"id": "x", "question": "q", "expect_tools_any": ["get_database_overview"], "contains_any": ["synthetic"]}
    assert evaluate_case(case, r)["passed"]
    r.answer = "The database holds 123456 cases."
    assert grounding_rate(r) == 0.0
