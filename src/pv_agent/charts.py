"""Render chart specs (produced by tools) as Plotly figures."""
from __future__ import annotations


def to_figure(chart: dict):
    import plotly.graph_objects as go  # lazy: plotly is an optional dependency

    fig = go.Figure()
    kind = chart["type"]
    for name, ys in chart["series"].items():
        if kind == "line":
            fig.add_trace(go.Scatter(x=chart["x"], y=ys, mode="lines+markers", name=name))
        else:
            fig.add_trace(go.Bar(x=chart["x"], y=ys, name=name))
    fig.update_layout(
        title=chart["title"], xaxis_title=chart.get("x_label", ""), yaxis_title=chart.get("y_label", ""),
        barmode="group", height=380, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=len(chart["series"]) > 1,
    )
    return fig
