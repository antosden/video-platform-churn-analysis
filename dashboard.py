import gradio as gr
from hypothesis.h4 import *



with gr.Blocks() as demo:
    with gr.Tab("Гипотеза 1"):
        gr.Markdown("")

    with gr.Tab("Гипотеза 2"):
        gr.Markdown("")

    with gr.Tab("Гипотеза 3"):
        gr.Markdown("")

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