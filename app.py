"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from pv_agent.agent import PharmacovigilanceAgent
from pv_agent.charts import to_figure
from pv_agent.config import get_settings

st.set_page_config(page_title="Pharmacovigilance AI Agent", layout="wide")
settings = get_settings()

EXAMPLES = [
    "What are the strongest safety signals for prednisone?",
    "Compare ibuprofen and atorvastatin - which reactions differ most?",
    "How have semaglutide reports changed over time?",
    "What share of metformin reports involve a serious outcome, and which outcomes?",
    "Which age groups report adalimumab most often?",
]


@st.cache_resource
def get_agent(db_path: str, model: str):
    from dataclasses import replace
    from pathlib import Path

    return PharmacovigilanceAgent(replace(settings, db_path=Path(db_path), model=model))


with st.sidebar:
    st.title("PV Agent")
    import os
    current_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    is_live_key = bool(current_key and current_key.lower() not in {"your-key-here", "none", "mock", "offline", "local", "free"})
    if is_live_key:
        st.success("Connected to Anthropic API")
    else:
        st.success("Free Local Mode Active")
        st.caption("Zero cost: runs locally using typed analysis tools and statistical signal detection without requiring any paid API key.")

    db_path = st.text_input("Database path", str(settings.db_path))
    model = st.text_input("Model", settings.model if is_live_key else "local-free-mode")
    
    with st.expander("API Key Configuration (Optional)"):
        user_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        if user_key:
            os.environ["ANTHROPIC_API_KEY"] = user_key
            st.rerun()

    st.divider()
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

st.header("Pharmacovigilance AI Agent")
try:
    agent = get_agent(db_path, model)
    overview = agent.tools.get_database_overview().data
    st.info(f"**Data source:** {overview['data_source']}  |  **Cases:** {overview['n_cases']:,}  |  "
            f"**Reports:** {overview['first_report']} to {overview['last_report']}")
except Exception as e:
    st.error(f"{e}\n\nBuild a database first: `pv-agent build-db --synthetic`")
    st.stop()

st.session_state.setdefault("messages", [])
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for ch in m.get("charts", []):
            st.plotly_chart(to_figure(ch), use_container_width=True)
        if m.get("trace"):
            with st.expander(f"Tool calls ({len(m['trace'])})"):
                for t in m["trace"]:
                    st.code(f"{t.name}({t.input})  # {t.duration_ms} ms" + (" ERROR" if t.is_error else ""), language="python")
                    st.json(t.output, expanded=False)

question = st.chat_input("Ask about adverse events, signals, outcomes, trends...") or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                r = agent.ask(question, history)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.markdown(r.answer)
        for ch in r.charts[-2:]:
            st.plotly_chart(to_figure(ch), use_container_width=True)
        st.caption(f"{r.steps} step(s) · {len(r.tool_calls)} tool call(s) · {r.input_tokens + r.output_tokens:,} tokens")
    st.session_state["messages"].append({"role": "assistant", "content": r.answer, "charts": r.charts[-2:], "trace": r.tool_calls})
