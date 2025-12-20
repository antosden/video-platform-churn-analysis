import numpy as np
import pandas as pd
import plotly.graph_objects as go

def _sort_city_tbl(tbl: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "avg_watch":
        return tbl.sort_values("avg_watch", ascending=False)
    if sort_by == "users":
        return tbl.sort_values("users", ascending=False)
    return tbl.sort_values("churn_rate", ascending=False)


def plot_city_overview(
    city_tbl: pd.DataFrame,
    top_n: int,
    sort_by: str,
    highlight_cities: list[str] | None,
):
    tbl = city_tbl.copy()
    tbl = _sort_city_tbl(tbl, sort_by)
    tbl = tbl.head(int(top_n)).copy()

    highlight_set = set(highlight_cities or [])

    watch_colors = ["#ff7f0e" if c in highlight_set else "#ffb55a" for c in tbl["city"]]
    churn_colors = ["#1f77b4" if c in highlight_set else "#8bb9ff" for c in tbl["city"]]

    hover_watch = (
        "<b>%{x}</b><br>"
        "Avg watch: %{y:.3f} min/day<br>"
        "Users: %{customdata[0]}<extra></extra>"
    )
    hover_churn = (
        "<b>%{x}</b><br>"
        "Churn rate: %{y:.2f}%<br>"
        "Users: %{customdata[0]}<extra></extra>"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tbl["city"],
            y=tbl["avg_watch"],
            name="Время просмотра (min/day)",
            marker_color=watch_colors,
            opacity=0.9,
            customdata=np.stack([tbl["users"]], axis=-1),
            hovertemplate=hover_watch,
            offsetgroup=0,
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Bar(
            x=tbl["city"],
            y=tbl["churn_rate"],
            name="Отток (%)",
            marker_color=churn_colors,
            opacity=0.9,
            customdata=np.stack([tbl["users"]], axis=-1),
            hovertemplate=hover_churn,
            offsetgroup=1,
            yaxis="y2",
        )
    )

    fig.update_layout(
        autosize=True,
        title="Города: Время просмотра (min/day) vs Отток (%)",
        barmode="group",
        xaxis=dict(title="Город", tickangle=-25),
        yaxis=dict(title="Время просмотра (min/day)"),
        yaxis2=dict(title="Отток (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=70, b=90),
        height=520,
    )

    show_tbl = tbl[["city", "avg_watch", "churn_rate", "users"]].copy()
    show_tbl["avg_watch"] = show_tbl["avg_watch"].round(3)
    show_tbl["churn_rate"] = show_tbl["churn_rate"].round(2)

    return fig


def compare_two_cities(city_tbl: pd.DataFrame, city_a: str, city_b: str):
    tbl = city_tbl.copy()

    a = tbl[tbl["city"] == city_a].head(1)
    b = tbl[tbl["city"] == city_b].head(1)

    if a.empty or b.empty:
        return "—", "—", "—", go.Figure(), pd.DataFrame()

    a = a.iloc[0]
    b = b.iloc[0]

    delta_watch = float(a["avg_watch"] - b["avg_watch"])
    delta_churn = float(a["churn_rate"] - b["churn_rate"])
    delta_users = int(a["users"] - b["users"])

    delta_watch_txt = f"{delta_watch:+.3f} min/day"
    delta_churn_txt = f"{delta_churn:+.2f} pp"
    delta_users_txt = f"{delta_users:+d}"

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[city_a, city_b],
            y=[a["avg_watch"], b["avg_watch"]],
            name="Время просмотра (min/day)",
            marker_color="#ff7f0e",
            opacity=0.85,
            offsetgroup=0,
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[city_a, city_b],
            y=[a["churn_rate"], b["churn_rate"]],
            name="Отток (%)",
            marker_color="#1f77b4",
            opacity=0.85,
            offsetgroup=1,
            yaxis="y2",
        )
    )

    fig.update_layout(
        autosize=True,
        title="Сравнить два города: Время просмотра (min/day) vs Отток (%)",
        barmode="group",
        xaxis=dict(title="Город"),
        yaxis=dict(title="Время просмотра (min/day)"),
        yaxis2=dict(title="Отток (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=70, b=80),
        height=420,
    )

    show_tbl = tbl[tbl["city"].isin([city_a, city_b])][["city", "avg_watch", "churn_rate", "users"]].copy()
    show_tbl["avg_watch"] = show_tbl["avg_watch"].round(3)
    show_tbl["churn_rate"] = show_tbl["churn_rate"].round(2)

    return delta_watch_txt, delta_churn_txt, delta_users_txt, fig, show_tbl