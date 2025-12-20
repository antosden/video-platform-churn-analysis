import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_threshold_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Делим пользователей на 2 группы по порогу среднего времени просмотра (мин/день)
    и считаем churn_rate, users, avg_watch, median_watch.
    """
    d = df.copy()
    d = d.dropna(subset=["avg_min_watch_daily", "churn"])

    low = d[d["avg_min_watch_daily"] < threshold]
    high = d[d["avg_min_watch_daily"] >= threshold]

    def _row(label: str, part: pd.DataFrame) -> dict:
        users = int(len(part))
        churn_rate = float(part["churn"].mean() * 100) if users else 0.0
        avg_watch = float(part["avg_min_watch_daily"].mean()) if users else np.nan
        med_watch = float(part["avg_min_watch_daily"].median()) if users else np.nan
        return {
            "group": label,
            "users": users,
            "churn_rate_%": churn_rate,
            "avg_watch": avg_watch,
            "median_watch": med_watch,
        }

    out = pd.DataFrame(
        [
            _row(f"< {threshold:.0f} мин/день", low),
            _row(f"≥ {threshold:.0f} мин/день", high),
        ]
    )

    # округления “для людей”
    out["churn_rate_%"] = out["churn_rate_%"].round(2)
    out["avg_watch"] = out["avg_watch"].round(3)
    out["median_watch"] = out["median_watch"].round(3)
    return out


def plot_threshold_churn(df: pd.DataFrame, threshold: float):
    """
    Возвращает (fig, table_df) для Gradio:
    - fig: барчарт churn rate по 2 группам (ниже/выше порога)
    - table_df: таблица с цифрами
    """
    tbl = build_threshold_table(df, threshold)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tbl["group"],
            y=tbl["churn_rate_%"],
            text=tbl["churn_rate_%"].astype(str) + "%",
            textposition="outside",
            name="Churn rate (%)",
        )
    )

    fig.update_layout(
        title=f"Отток по порогу времени просмотра (порог = {threshold:.0f} мин/день)",
        xaxis_title="Группа по времени просмотра",
        yaxis_title="Churn rate (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=40, t=70, b=70),
        height=420,
        showlegend=False,
    )

    return fig, tbl