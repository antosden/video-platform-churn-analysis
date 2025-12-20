import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from hypothesis.h4 import *
from hypothesis.h2_3 import *



with gr.Blocks() as demo:
    with gr.Tab("Гипотеза 1"):
        gr.Markdown("")

    with gr.Tab("Гипотеза 2"):
        with gr.Tab("Гипотеза 2"):
            run_gradio_app(df, initial_threshold=30)

        with gr.Tab("Гипотеза 2.1 и 2.2"):

            run_categorical_analysis_app(df)


    with gr.Tab("Гипотеза 3"):
        gr.Markdown("")
        gr.Markdown("# Анализ порога вовлеченности пользователей")
        gr.Markdown("Настройте параметры для анализа взаимосвязи времени просмотра и конверсии")

        with gr.Row():
            with gr.Column(scale=1):
                threshold = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Порог вовлеченности (минут в день)"
                )

                min_days = gr.Slider(
                    minimum=1, maximum=7, value=4, step=1,
                   label="Минимальное дней активности"
                )

                days_to_compare = gr.CheckboxGroup(
                    choices=[3, 4, 5, 6, 7],
                    value=[4, 5, 6],
                    label="Дни для сравнения"
                )

                analyze_btn = gr.Button("Проанализировать", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Визуализация результатов")
                with gr.Tab("Конверсия по группам"):
                    plot1 = gr.Plot(label="Конверсия по порогу вовлеченности")

                with gr.Tab("Распределение времени просмотра"):
                    plot2 = gr.Plot(label="Распределение вовлеченности по группам")

                with gr.Tab("Статистика по дням"):
                    plot3 = gr.Plot(label="Метрики по дням активности")

        analyze_btn.click(
            fn=create_engagement_analysis,
            inputs=[threshold, min_days, days_to_compare],
            outputs=[plot1, plot2, plot3]
        )

        demo.launch(share=True)
    with gr.Tab("Гипотеза 4"):
        gr.Markdown("## Гипотеза 4: География <-> просмотр <-> отток")
        with gr.Row():
            top_n = gr.Slider(3, CITY_COUNT, value=10, step=1, label="Топ городов по просмотрам")
            sort_by = gr.Radio(
                choices=["churn_rate_%", "avg_watch", "users"],
                value="churn_rate_%",
                label="Sort by"
            )

        btn = gr.Button("Применить")

        plot = gr.Plot(label="Топ городов")
        table = gr.Dataframe(interactive=False, label="Топ городов")

        btn.click(
            fn=plot_city_bars_plotly,
            inputs=[top_n, sort_by],
            outputs=[plot, table],
        )

        demo.load(
            fn=plot_city_bars_plotly,
            inputs=[top_n, sort_by],
            outputs=[plot, table],
        )

        gr.Markdown("---")
        gr.Markdown("### Сравнение двух городов")

        with gr.Row():
            city_a = gr.Dropdown(choices=ALL_CITIES, value="Уфа" if "Уфа" in ALL_CITIES else ALL_CITIES[0], label="City A")
            city_b = gr.Dropdown(choices=ALL_CITIES, value="Екатеринбург" if "Екатеринбург" in ALL_CITIES else ALL_CITIES[min(1, len(ALL_CITIES)-1)], label="City B")

        with gr.Row():
            kpi_watch = gr.Textbox(label="delta Avg watch (A - B)", value="—", interactive=False)
            kpi_churn = gr.Textbox(label="delta Churn rate (A - B)", value="—", interactive=False)
            kpi_users = gr.Textbox(label="delta Users (A - B)", value="—", interactive=False)

        cmp_plot = gr.Plot(label="Comparison chart")
        cmp_table = gr.Dataframe(interactive=False, label="Comparison table")

        city_a.change(
            fn=compare_two_cities_plotly,
            inputs=[city_a, city_b],
            outputs=[cmp_plot, cmp_table, kpi_watch, kpi_churn, kpi_users],
        )
        city_b.change(
            fn=compare_two_cities_plotly,
            inputs=[city_a, city_b],
            outputs=[cmp_plot, cmp_table, kpi_watch, kpi_churn, kpi_users],
        )


demo.launch()
