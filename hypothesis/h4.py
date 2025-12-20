import numpy as np
import plotly.graph_objects as go
import pandas as pd


df = pd.read_csv("videostreaming_platform.csv")

# минимальная чистка
df["city"] = df["city"].fillna("Unknown")
df["avg_min_watch_daily"] = pd.to_numeric(df["avg_min_watch_daily"], errors="coerce")
df["churn"] = pd.to_numeric(df["churn"], errors="coerce")

def build_city_table(data: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        data.groupby("city", as_index=False)
        .agg(
            avg_watch=("avg_min_watch_daily", "mean"),
            churn_rate=("churn", "mean"),
            users=("user_id", "count"),
        )
    )
    tbl["churn_rate_%"] = tbl["churn_rate"] * 100
    return tbl



CITY_TBL = build_city_table(df)
ALL_CITIES = sorted(CITY_TBL["city"].astype(str).unique().tolist())
CITY_COUNT = CITY_TBL["city"].nunique()

def plot_city_bars_plotly(
    top_n: int = 10,
    sort_by: str = "churn_rate_%",
):
    tbl = CITY_TBL.copy()

    # сортировка
    if sort_by == "avg_watch":
        tbl = tbl.sort_values("avg_watch", ascending=False)
    elif sort_by == "users":
        tbl = tbl.sort_values("users", ascending=False)
    else:
        tbl = tbl.sort_values("churn_rate_%", ascending=False)

    tbl = tbl.head(int(top_n)).copy()

    watch_colors = ["#ff7f0e"] * len(tbl)
    churn_colors = ["#1f77b4"] * len(tbl)

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

    # Avg watch (левая ось)
    fig.add_trace(
        go.Bar(
            x=tbl["city"],
            y=tbl["avg_watch"],
            name="Avg watch (min/day)",
            marker_color=watch_colors,
            opacity=0.9,
            customdata=np.stack([tbl["users"]], axis=-1),
            hovertemplate=hover_watch,
            offsetgroup=0,
            yaxis="y",
        )
    )

    # Churn rate % (правая ось)
    fig.add_trace(
        go.Bar(
            x=tbl["city"],
            y=tbl["churn_rate_%"],
            name="Churn rate (%)",
            marker_color=churn_colors,
            opacity=0.9,
            customdata=np.stack([tbl["users"]], axis=-1),
            hovertemplate=hover_churn,
            offsetgroup=1,
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Города: Avg watch time vs Churn rate",
        barmode="group",
        xaxis=dict(title="City", tickangle=-25),
        yaxis=dict(title="Avg watch (min/day)"),
        yaxis2=dict(title="Churn rate (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=70, b=90),
        height=520,
    )

    show_tbl = tbl[["city", "avg_watch", "churn_rate_%", "users"]].copy()
    show_tbl["avg_watch"] = show_tbl["avg_watch"].round(3)
    show_tbl["churn_rate_%"] = show_tbl["churn_rate_%"].round(2)

    return fig, show_tbl


def compare_two_cities_plotly(city_a: str, city_b: str):
    tbl = CITY_TBL.copy()
    tbl["city"] = tbl["city"].astype(str)

    pick = tbl[tbl["city"].isin([city_a, city_b])].copy().drop_duplicates(subset=["city"])

    # KPI / table base
    show_tbl = pick[["city", "avg_watch", "churn_rate_%", "users"]].copy()
    show_tbl["avg_watch"] = show_tbl["avg_watch"].round(3)
    show_tbl["churn_rate_%"] = show_tbl["churn_rate_%"].round(2)

    # KPI deltas
    delta_watch = "—"
    delta_churn = "—"
    delta_users = "—"
    if len(show_tbl) == 2:
        a = show_tbl[show_tbl["city"] == city_a].iloc[0]
        b = show_tbl[show_tbl["city"] == city_b].iloc[0]
        delta_watch = f"{float(a['avg_watch'] - b['avg_watch']):+.3f} min/day"
        delta_churn = f"{float(a['churn_rate_%'] - b['churn_rate_%']):+.2f} pp"
        delta_users = f"{int(a['users'] - b['users']):+d}"


    watch_colors = "#ffb55a"
    churn_colors = "#8bb9ff"

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

    fig.add_trace(go.Bar(
        x=pick["city"],
        y=pick["avg_watch"],
        name="Avg watch (min/day)",
        marker_color=watch_colors,
        opacity=0.9,
        customdata=np.stack([pick["users"]], axis=-1),
        hovertemplate=hover_watch,
        offsetgroup=0,
        yaxis="y",
    ))

    fig.add_trace(go.Bar(
        x=pick["city"],
        y=pick["churn_rate_%"],
        name="Churn rate (%)",
        marker_color=churn_colors,
        opacity=0.9,
        customdata=np.stack([pick["users"]], axis=-1),
        hovertemplate=hover_churn,
        offsetgroup=1,
        yaxis="y2",
    ))

    fig.update_layout(
        title="Сравнить два города: Avg watch vs Churn",
        barmode="group",
        xaxis=dict(title="City"),
        yaxis=dict(title="Avg watch (min/day)"),
        yaxis2=dict(title="Churn rate (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=70, b=60),
        height=430,
    )

    return fig, show_tbl, delta_watch, delta_churn, delta_users


def swap_cities(a: str, b: str):
    return b, a