import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import plotly.graph_objects as go


def calc_phi_results(
    data_encoded: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    results = []

    for col in data_encoded.columns:
        if any(cat in col for cat in ['city_', 'device_', 'source_', 'favourite_genre_']):
            conf_matrix = pd.crosstab(data_encoded[col], data_encoded['churn'])

            # Фи-коэффициент для 2x2 таблицы
            if conf_matrix.shape == (2, 2):
                a, b, c, d = conf_matrix.values.flatten()
                phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            else:
                # V Крамера для таблиц больше 2x2
                chi2 = chi2_contingency(conf_matrix)[0]
                n = conf_matrix.sum().sum()
                phi = np.sqrt(chi2 / n)

            # p-value для проверки значимости
            _, p_value, _, _ = chi2_contingency(conf_matrix)

            results.append({
                'feature': col,
                'phi_coefficient': round(phi, 4),
                'p_value': round(p_value, 6),
                'significant': p_value < 0.05
            })

    phi_results = pd.DataFrame(results)

    if not phi_results.empty:
        phi_results = phi_results.sort_values(
            "phi_coefficient", key=lambda s: s.abs(), ascending=False
        )

    return phi_results[['feature', 'phi_coefficient', 'p_value', 'significant']]


def plot_phi_bar(phi_df: pd.DataFrame, top_k: int = 20) -> go.Figure:
    if phi_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Нет данных для отображения")
        return fig

    df = phi_df.head(top_k).copy()
    df = df.sort_values("phi_coefficient", key=lambda s: s.abs(), ascending=True)

    colors = df["significant"].map({True: "#d62728", False: "#7f7f7f"})

    fig = go.Figure(
        go.Bar(
            x=df["phi_coefficient"].abs(),
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "|φ| = %{x:.4f}<br>"
                "p-value = %{customdata[0]:.6f}<br>"
                "significant = %{customdata[1]}<extra></extra>"
            ),
            customdata=df[["p_value", "significant"]],
        )
    )

    fig.update_layout(
        title="Сила связи категориальных признаков с оттоком (|φ|)",
        xaxis_title="|φ| / V Cramér",
        yaxis_title="Признак",
        margin=dict(l=260, r=40, t=60, b=60),
        height=520,
    )

    return fig