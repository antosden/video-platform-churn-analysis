import gradio as gr
import pandas as pd
import numpy as np

from hypothesis.dataset import prepare_dataset
from hypothesis.h0 import calc_phi_results, plot_phi_bar
from hypothesis.h1 import plot_threshold_churn
from hypothesis.h4 import plot_city_overview, compare_two_cities


def main():
    df, city_tbl, all_cities, city_count = prepare_dataset()

    with gr.Blocks(title="VideoStreaming Dashboard") as demo:
        st_df = gr.State(df)
        st_city_tbl = gr.State(city_tbl)

        with gr.Tabs():
            with gr.TabItem("Гипотеза 0"):
                gr.Markdown("## Категориальные признаки ↔ отток")

                with gr.Row():
                    with gr.Column(scale=2, min_width=720):
                        with gr.Tabs():
                            with gr.TabItem("Сила связи (φ)"):
                                phi_plot = gr.Plot()

                            with gr.TabItem("Таблица значений"):
                                phi_table = gr.Dataframe(interactive=False)

                def render_h0(df_in: pd.DataFrame):
                    data_encoded = pd.get_dummies(
                        df_in,
                        columns=["city", "device", "source", "favourite_genre"],
                        drop_first=False,
                    )

                    phi_df = calc_phi_results(data_encoded, alpha=0.05)
                    fig = plot_phi_bar(phi_df, top_k=25)

                    return fig, phi_df

                demo.load(
                    fn=render_h0,
                    inputs=[st_df],
                    outputs=[phi_plot, phi_table],
                )

            with gr.TabItem("Гипотеза 1"):
                default_thr = float(np.nanmedian(df["avg_min_watch_daily"])) if len(df) else 10.0
                default_thr = max(1.0, round(default_thr))

                max_thr = float(np.nanmax(df["avg_min_watch_daily"])) if len(df) else 60.0
                max_thr = max(10.0, round(max_thr))
                gr.Markdown("## Гипотеза 1: Время просмотра ↔ отток/подписка")

                threshold = gr.Slider(
                    minimum=1,
                    maximum=max_thr,
                    value=min(default_thr, max_thr),
                    step=1,
                    label="Порог среднего времени просмотра (мин/день)",
                )

                h1_plot = gr.Plot(label="Churn rate по группам")
                h1_table = gr.Dataframe(interactive=False, label="Таблица по группам")

                def do_h1(df_in, thr):
                    return plot_threshold_churn(df_in, float(thr))
                demo.load(
                    fn=do_h1,
                    inputs=[st_df, threshold],
                    outputs=[h1_plot, h1_table],
                )

                threshold.change(
                    fn=do_h1,
                    inputs=[st_df, threshold],
                    outputs=[h1_plot, h1_table],
                )

            with gr.TabItem("Гипотеза 2"):
                gr.Markdown("TODO: Valentina")

            with gr.TabItem("Гипотеза 3"):
                gr.Markdown("TODO: Valentina")

            with gr.TabItem("Гипотеза 4"):
                gr.Markdown("## География ↔ просмотр ↔ отток")

                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        top_n = gr.Slider(
                            minimum=3,
                            maximum=city_count,
                            value=min(10, city_count),
                            step=1,
                            label="Количество городов",
                        )

                        sort_by = gr.Radio(
                            choices=["% оттока", "Среднее время просмотра", "Пользователи"],
                            value="% оттока",
                            label="Сортировка",
                        )

                        highlight_cities = gr.Dropdown(
                            choices=all_cities,
                            value=[],
                            multiselect=True,
                            label="Подсветить города",
                        )

                        gr.Markdown("### Сравнение городов")

                        city_a = gr.Dropdown(
                            choices=all_cities,
                            value=all_cities[0],
                            label="Город A",
                        )
                        city_b = gr.Dropdown(
                            choices=all_cities,
                            value=all_cities[1],
                            label="Город B",
                        )

                    with gr.Column(scale=2, min_width=720):
                        with gr.Tabs():
                            with gr.TabItem("Обзор"):
                                overview_plot = gr.Plot()

                            with gr.TabItem("Сравнение"):
                                with gr.Row():
                                    delta_watch = gr.Textbox(label="Δ Avg watch (A − B)")
                                    delta_churn = gr.Textbox(label="Δ Churn % (A − B)")
                                    delta_users = gr.Textbox(label="Δ Users (A − B)")

                                compare_plot = gr.Plot()

                def update_overview(tbl, n, sort, highlight):
                    return plot_city_overview(
                        city_tbl=tbl,
                        top_n=int(n),
                        sort_by=str(sort),
                        highlight_cities=highlight or [],
                    )

                def update_compare(tbl, a, b):
                    return compare_two_cities(tbl, str(a), str(b))

                demo.load(
                    fn=update_overview,
                    inputs=[st_city_tbl, top_n, sort_by, highlight_cities],
                    outputs=[overview_plot],
                )

                top_n.change(update_overview, [st_city_tbl, top_n, sort_by, highlight_cities], overview_plot)
                sort_by.change(update_overview, [st_city_tbl, top_n, sort_by, highlight_cities], overview_plot)
                highlight_cities.change(update_overview, [st_city_tbl, top_n, sort_by, highlight_cities], overview_plot)

                demo.load(
                    fn=update_compare,
                    inputs=[st_city_tbl, city_a, city_b],
                    outputs=[delta_watch, delta_churn, delta_users, compare_plot],
                )

                city_a.change(update_compare, [st_city_tbl, city_a, city_b],
                              [delta_watch, delta_churn, delta_users, compare_plot])
                city_b.change(update_compare, [st_city_tbl, city_a, city_b],
                              [delta_watch, delta_churn, delta_users, compare_plot])

        demo.launch()


if __name__ == "__main__":
    main()