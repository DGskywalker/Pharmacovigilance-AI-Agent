"""Command-line interface: build data, ask questions, chat, call tools directly, run evals."""
from __future__ import annotations

import argparse
import json
import sys

from .agent import PharmacovigilanceAgent, trace_to_text
from .config import ROOT, get_settings


def _settings(args):
    s = get_settings()
    if getattr(args, "db", None):
        from dataclasses import replace
        from pathlib import Path

        s = replace(s, db_path=Path(args.db))
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pv-agent", description="Pharmacovigilance AI Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-db", help="Build the SQLite database (synthetic or real FAERS).")
    b.add_argument("--synthetic", action="store_true", help="Generate synthetic FAERS-style data.")
    b.add_argument("--faers-dir", help="Folder containing extracted FAERS ASCII quarterly files.")
    b.add_argument("--cases", type=int, default=25_000)
    b.add_argument("--db")

    a = sub.add_parser("ask", help="Ask one question.")
    a.add_argument("question")
    a.add_argument("--db")
    a.add_argument("-v", "--verbose", action="store_true", help="Show the tool-call trace.")

    c = sub.add_parser("chat", help="Interactive chat session.")
    c.add_argument("--db")

    t = sub.add_parser("tool", help="Call an analysis tool directly (no LLM / API key needed).")
    t.add_argument("name")
    t.add_argument("args", nargs="?", default="{}", help='JSON arguments, e.g. \'{"drug":"prednisone"}\'')
    t.add_argument("--db")

    e = sub.add_parser("eval", help="Run the evaluation suite against the live agent.")
    e.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    e.add_argument("--out", default=str(ROOT / "evals" / "results.json"))
    e.add_argument("--db")

    args = p.parse_args(argv)
    settings = _settings(args)

    if args.cmd == "build-db":
        if args.faers_dir:
            from .loader import build_from_faers

            path = build_from_faers(args.faers_dir, args.db or settings.db_path)
        elif args.synthetic:
            from .synthetic import build_synthetic_db

            path = build_synthetic_db(args.db or settings.db_path, n_cases=args.cases)
        else:
            p.error("Provide --synthetic or --faers-dir.")
        print(f"Database written to {path}")
        return 0

    if args.cmd == "tool":
        from .tools import PharmacovigilanceTools, ToolError

        tools = PharmacovigilanceTools(settings.db_path)
        try:
            print(json.dumps(tools.call(args.name, json.loads(args.args)).data, indent=2, default=str))
        except ToolError as err:
            print(f"Tool error: {err}", file=sys.stderr)
            return 1
        return 0

    agent = PharmacovigilanceAgent(settings)

    if args.cmd == "ask":
        r = agent.ask(args.question)
        if args.verbose:
            print(trace_to_text(r), "\n")
        print(r.answer)
        return 0

    if args.cmd == "chat":
        history: list[dict] = []
        print("Pharmacovigilance AI Agent - type 'exit' to quit.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            r = agent.ask(q, history)
            print(f"\nagent> {r.answer}")
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]
        return 0

    if args.cmd == "eval":
        from .evals import run_eval

        report = run_eval(agent, args.questions, args.out)
        print(json.dumps(report["summary"], indent=2))
        for r in report["results"]:
            print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']}: {r['question']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
