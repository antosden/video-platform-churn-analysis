import gradio as gr
import pandas as pd
import numpy as np

from hypothesis.dataset import prepare_dataset
from hypothesis.h0 import calc_phi_results, plot_phi_bar
from hypothesis.h1 import plot_threshold_churn
from hypothesis.h2 import analyze_threshold, analyze_categorical_features
from hypothesis.h3 import create_engagement_analysis
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
                with gr.Tab("Гипотеза 2"):
                    print(f"Загружено {len(df):,} записей")
                    initial_threshold = 30
                    # Находим разумные границы для слайдера
                    min_time = int(max(1, df['avg_min_watch_daily'].min()))
                    max_time = int(df['avg_min_watch_daily'].max())

                    # Определяем оптимальные шаги для слайдера
                    if max_time - min_time > 100:
                        step = 5
                    elif max_time - min_time > 50:
                        step = 2
                    else:
                        step = 1

                    gr.Markdown("# Анализ влияния времени просмотра на отток пользователей")
                    gr.Markdown(
                        f"**Всего данных:** {len(df):,} пользователей | **Общий отток:** {df['churn'].mean():.1%}")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Настройки порога")
                            threshold_slider = gr.Slider(
                                minimum=min_time,
                                maximum=max_time,
                                value=initial_threshold,
                                step=step,
                                label="Порог времени просмотра (минут)",
                                info=f"Диапазон: {min_time}-{max_time} мин"
                            )

                            # Быстрые кнопки для часто используемых значений
                            gr.Markdown("### Быстрый выбор:")
                            quick_buttons = gr.Row()
                            with quick_buttons:
                                for value in [5, 10, 15, 30, 60]:
                                    if min_time <= value <= max_time:
                                        gr.Button(f"{value} мин", size="sm").click(
                                            fn=lambda x=value: x,
                                            outputs=threshold_slider
                                        )

                            gr.Markdown("### Ключевые метрики")
                            z_stat_output = gr.Textbox(label="Z-статистика", interactive=False)
                            p_value_output = gr.Textbox(label="p-value", interactive=False)
                            rr_output = gr.Textbox(label="Относительный риск", interactive=False)
                            significance_output = gr.Textbox(label="Статистическая значимость", interactive=False)

                        with gr.Column(scale=2):
                            gr.Markdown("### Визуализация результатов")
                            with gr.Tab("Процент оттока"):
                                plot1 = gr.Plot(label="Процент оттока по группам")

                            with gr.Tab("Доверительные интервалы"):
                                plot2 = gr.Plot(label="Доверительные интервалы (95%)")

                    gr.Markdown("### Подробные результаты")
                    results_text = gr.Markdown()

                    # Функция-обертка для обновления
                    def update_wrapper(threshold):
                        return analyze_threshold(threshold, df)

                    # Обработчик изменения порога
                    threshold_slider.change(
                        fn=update_wrapper,
                        inputs=threshold_slider,
                        outputs=[plot1, plot2, results_text, z_stat_output, p_value_output, rr_output,
                                 significance_output]
                    )

                    # Инициализируем с начальным значением
                    demo.load(
                        fn=lambda: update_wrapper(initial_threshold),
                        outputs=[plot1, plot2, results_text, z_stat_output, p_value_output, rr_output,
                                 significance_output]
                    )

                    gr.Markdown("---")
                    gr.Markdown("### ℹКак интерпретировать результаты:")
                    gr.Markdown("""
                            - **p-value < 0.05**: разница статистически значима
                            - **Относительный риск > 1**: меньший просмотр = больший риск оттока
                            - **Доверительные интервалы не пересекаются**: подтверждение значимости
                            """)
                with gr.Tab("Гипотеза 2.1 и 2.2"):
                    max_time = int(df['avg_min_watch_daily'].max())
                    median_time = int(df['avg_min_watch_daily'].median())

                    # Проверяем наличие категориальных колонок
                    cat_columns = ['city', 'device', 'source', 'favourite_genre']
                    available_columns = [col for col in cat_columns if col in df.columns]

                    print(f"Доступные колонки: {available_columns}")

                    gr.Markdown("# Анализ влияния категорий на отток активных пользователей")
                    gr.Markdown(
                        "Исследуем, как город, устройство, источник трафика и любимый жанр влияют на отток среди пользователей с высоким временем просмотра")

                    # Информация о данных
                    gr.Markdown(f"""
                        ### Информация о данных:
                        - **Всего пользователей:** {len(df):,}
                        - **Среднее время просмотра:** {df['avg_min_watch_daily'].mean():.1f} мин
                        - **Общий отток:** {df['churn'].mean():.1%}
                        - **Доступные категории:** {', '.join(available_columns) if available_columns else 'НЕТ'}
                        """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Параметры анализа")

                            threshold = gr.Slider(
                                minimum=1,
                                maximum=min(120, max_time),
                                value=min(30, median_time),
                                step=1,
                                label="Минимальное время просмотра (минут)",
                                info=f"Анализируются только пользователи с просмотром больше этого значения"
                            )

                            p_value = gr.Slider(
                                minimum=0.001,
                                maximum=0.1,
                                value=0.05,
                                step=0.001,
                                label="Порог статистической значимости",
                                info="p-value для определения значимости"
                            )

                            gr.Markdown("### Быстрые настройки")
                            quick_buttons = gr.Row()
                            with quick_buttons:
                                for val in [15, 30, 45, 60]:
                                    if val <= max_time:
                                        gr.Button(f"{val} мин", size="sm").click(
                                            fn=lambda x=val: x,
                                            outputs=threshold
                                        )

                            analyze_btn = gr.Button("Запустить анализ", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            gr.Markdown("### Результаты анализа")

                            with gr.Tab("Сила связи признаков"):
                                plot1 = gr.Plot(label="Cramer's V по признакам")

                            with gr.Tab("Отток по категориям"):
                                plot2 = gr.Plot(label="Детали по категориям")

                            with gr.Tab("Подробный отчет"):
                                report_output = gr.Markdown(label="Аналитический отчет")

                    def analyze(threshold_val, p_value_val):
                        try:
                            fig1, fig2, report = analyze_categorical_features(
                                df, threshold_val, p_value_val
                            )
                            return fig1, fig2, report
                        except Exception as e:
                            print(f"Ошибка: {str(e)}")

                    # Подключаем обработчики
                    analyze_btn.click(
                        fn=analyze,
                        inputs=[threshold, p_value],
                        outputs=[plot1, plot2, report_output]
                    )

                    # Автозапуск при изменении параметров
                    threshold.change(
                        fn=analyze,
                        inputs=[threshold, p_value],
                        outputs=[plot1, plot2, report_output]
                    )

                    p_value.change(
                        fn=analyze,
                        inputs=[threshold, p_value],
                        outputs=[plot1, plot2, report_output]
                    )

                    # Инициализация при загрузке
                    demo.load(
                        fn=lambda: analyze(min(30, median_time), 0.05),
                        outputs=[plot1, plot2, report_output]
                    )

                    gr.Markdown("---")
                    gr.Markdown("""
                        ### Методология анализа:

                        1. **Фильтрация**: Анализируются только пользователи с просмотром выше порога
                        2. **Статистический тест**: Хи-квадрат тест для таблиц сопряженности
                        3. **Мера связи**: Cramer's V (аналог Phi для таблиц больше 2×2)
                        4. **Значимость**: p-value для проверки статистической значимости

                        ### Как использовать результаты:
                        - **Признаки с высоким Cramer's V**: Сильно влияют на отток
                        - **Значимые признаки (p < 0.05)**: Связь статистически подтверждена
                        - **Категории с высоким оттоком**: Требуют внимания и улучшения
                        """)

            with gr.TabItem("Гипотеза 3"):
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
                    inputs=[st_df, threshold, min_days, days_to_compare],
                    outputs=[plot1, plot2, plot3]
                )

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
