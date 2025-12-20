import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from io import StringIO
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Гипотеза 4
def create_engagement_analysis(threshold_minutes, min_days_active, comparison_days):
    """
    Анализ порога вовлеченности с интерактивными параметрами
    """
    # Проверяем, что comparison_days не пустой
    if not comparison_days:
        comparison_days = [min_days_active]

    # Фильтруем пользователей с минимальным количеством дней
    filtered_df = df[df['number_of_days_logged'] >= min_days_active].copy()

    # Создаем группы по порогу
    filtered_df['above_threshold'] = filtered_df['avg_min_watch_daily'] >= threshold_minutes
    filtered_df['converted'] = filtered_df['churn'] == 0

    # Анализ конверсии
    group_analysis = filtered_df.groupby(['number_of_days_logged', 'above_threshold']).agg({
        'user_id': 'count',
        'churn': lambda x: (1 - x.mean()) * 100
    }).reset_index()

    # Переименовываем колонки
    group_analysis.columns = ['days_active', 'above_threshold', 'user_count', 'conversion_rate']

    # Создаем фигуры
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # 1. Конверсия по группам
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_days)))

    # Подготовка данных для графика
    bar_width = 0.8 / len(comparison_days)

    for idx, days in enumerate(sorted(comparison_days)):
        subset = group_analysis[group_analysis['days_active'] == days]
        if len(subset) == 2:  # есть обе группы
            # Данные для ниже/выше порога
            below_data = subset[subset['above_threshold'] == False]
            above_data = subset[subset['above_threshold'] == True]

            below_rate = below_data['conversion_rate'].iloc[0] if len(below_data) > 0 else 0
            above_rate = above_data['conversion_rate'].iloc[0] if len(above_data) > 0 else 0

            # Позиции для столбцов
            below_pos = idx * bar_width
            above_pos = 1 + idx * bar_width

            # Рисуем столбцы
            ax1.bar(below_pos, below_rate, width=bar_width*0.8,
                   color='red', alpha=0.7, label=f'{days} дн.' if idx == 0 else "")
            ax1.bar(above_pos, above_rate, width=bar_width*0.8,
                   color='green', alpha=0.7)

            # Добавляем подписи
            ax1.text(below_pos, below_rate + 1, f'{below_rate:.1f}%',
                    ha='center', fontsize=9)
            ax1.text(above_pos, above_rate + 1, f'{above_rate:.1f}%',
                    ha='center', fontsize=9)

    ax1.set_xlabel('Группа')
    ax1.set_ylabel('Конверсия, %')
    ax1.set_title(f'Конверсия по порогу вовлеченности ({threshold_minutes} мин/день)')
    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(['Ниже порога', 'Выше порога'])
    ax1.legend(title='Дней активности')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(group_analysis['conversion_rate'].max() * 1.2, 100))

    plt.tight_layout()
    fig1.canvas.draw()

    # 2. Распределение времени просмотра
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    data_to_plot = []
    labels_to_plot = []

    for days in sorted(comparison_days):
        for converted in [0, 1]:
            subset = filtered_df[(filtered_df['number_of_days_logged'] == days) &
                                (filtered_df['churn'] == converted)]
            if len(subset) > 0:
                data_to_plot.append(subset['avg_min_watch_daily'].dropna())
                status = "✓" if converted == 0 else "✗"
                labels_to_plot.append(f'{days}д. {"Удерж." if converted == 0 else "Отток"}')

    if data_to_plot:  # проверяем, что есть данные
        boxplot = ax2.boxplot(data_to_plot, widths=0.6, patch_artist=True)
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
        for patch, color in zip(boxplot['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.axhline(y=threshold_minutes, color='red', linestyle='--',
                   label=f'Порог: {threshold_minutes} мин', linewidth=2)
        ax2.set_xlabel('Группа')
        ax2.set_ylabel('Время просмотра, мин/день')
        ax2.set_title('Распределение времени просмотра по группам')
        ax2.set_xticks(range(1, len(labels_to_plot) + 1))
        ax2.set_xticklabels(labels_to_plot, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Нет данных для выбранных параметров',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    fig2.canvas.draw()

    # 3. Статистика по дням
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    days_stats = filtered_df.groupby('number_of_days_logged').agg({
        'user_id': 'count',
        'converted': 'mean',
        'avg_min_watch_daily': 'mean'
    }).reset_index()

    # Создаем второй оси Y
    ax3_secondary = ax3.twinx()

    # Столбцы для времени просмотра (основная ось Y слева)
    bars = ax3.bar(days_stats['number_of_days_logged'],
                   days_stats['avg_min_watch_daily'],
                   alpha=0.5, color='skyblue', label='Время просмотра (мин)',
                   width=0.6)

    # Линия для конверсии (вторичная ось Y справа)
    line = ax3_secondary.plot(days_stats['number_of_days_logged'],
                             days_stats['converted'] * 100,
                             color='green', marker='o', linewidth=2,
                             label='Конверсия (%)')

    ax3.set_xlabel('Дней активности')
    ax3.set_ylabel('Среднее время просмотра (мин)', color='skyblue')
    ax3_secondary.set_ylabel('Конверсия (%)', color='green')

    # Цвета меток осей
    ax3.tick_params(axis='y', labelcolor='skyblue')
    ax3_secondary.tick_params(axis='y', labelcolor='green')

    ax3.set_title('Метрики по дням активности')

    # Объединяем легенды
    lines_labels = [ax3.get_legend_handles_labels()[0][0],
                   ax3_secondary.get_legend_handles_labels()[0][0]]
    labels = [ax3.get_legend_handles_labels()[1][0],
             ax3_secondary.get_legend_handles_labels()[1][0]]

    ax3.legend(lines_labels, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig3.canvas.draw()

    return fig1, fig2, fig3





# Гипотеза 2.1 и 2.2
def analyze_categorical_features(df, threshold_minutes=30, p_value_threshold=0.05):
    """
    Анализирует влияние категориальных признаков на отток среди пользователей с просмотром > threshold
    """
    # Фильтруем пользователей с просмотром выше порога
    filtered_df = df[df['avg_min_watch_daily'] > threshold_minutes].copy()

    if len(filtered_df) < 50:
        error_msg = f"Слишком мало пользователей ({len(filtered_df)}) для анализа с порогом {threshold_minutes} мин"
        print(error_msg)
        return None, None, None, error_msg

    # Список категориальных колонок
    cat_columns = ['city', 'device', 'source', 'favourite_genre']

    # Проверяем, какие колонки существуют
    available_columns = [col for col in cat_columns if col in df.columns]

    if not available_columns:
        print("Не найдены категориальные колонки (city, device, source, favourite_genre)")
        return None, None, None, error_msg

    results = []
    detailed_analysis = []

    for col in cat_columns:

        try:
            # Создаем таблицу сопряженности
            conf_matrix = pd.crosstab(filtered_df[col], filtered_df['churn'])

            # Удаляем категории с малым количеством наблюдений
            min_category_size = 10
            category_sizes = conf_matrix.sum(axis=1)
            valid_categories = category_sizes[category_sizes >= min_category_size].index
            conf_matrix = conf_matrix.loc[valid_categories]

            if len(conf_matrix) < 2:
                continue

            # Хи-квадрат тест
            chi2, p_value, dof, expected = chi2_contingency(conf_matrix)

            # Рассчитываем размер эффекта (Cramer's V)
            n = conf_matrix.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * (min(conf_matrix.shape) - 1)))

            # Рассчитываем отток по категориям
            churn_rates = conf_matrix[1] / conf_matrix.sum(axis=1)

            # Находим максимальную разницу в оттоке между категориями
            if len(churn_rates) >= 2:
                churn_diff = churn_rates.max() - churn_rates.min()
                highest_churn_cat = churn_rates.idxmax()
                lowest_churn_cat = churn_rates.idxmin()
            else:
                churn_diff = 0
                highest_churn_cat = churn_rates.index[0] if len(churn_rates) > 0 else None
                lowest_churn_cat = highest_churn_cat

            results.append({
                'feature': col,
                'cramers_v': cramer_v,
                'p_value': p_value,
                'chi2': chi2,
                'churn_diff': churn_diff,
                'significant': p_value < p_value_threshold,
                'n_categories': len(conf_matrix),
                'total_users': n,
                'highest_churn_category': highest_churn_cat,
                'lowest_churn_category': lowest_churn_cat,
                'highest_churn_rate': churn_rates.max() if len(churn_rates) > 0 else 0,
                'lowest_churn_rate': churn_rates.min() if len(churn_rates) > 0 else 0
            })

            # Сохраняем детальные данные для визуализации
            detailed_analysis.append({
                'feature': col,
                'conf_matrix': conf_matrix,
                'churn_rates': churn_rates,
                'category_counts': conf_matrix.sum(axis=1)
            })

        except Exception as e:
            print(f"  Ошибка при анализе {col}: {str(e)}")
            continue

    if not results:
        print("Не удалось рассчитать статистики ни для одной категориальной колонки")
        return None, None, None, error_msg

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)

    # Сортируем по силе связи (Cramer's V)
    results_df = results_df.sort_values('cramers_v', ascending=False)

    # Создаем визуализации

    # 1. Bar chart с Cramer's V для всех признаков
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['green' if sig else 'red' for sig in results_df['significant']]
    bars = ax1.barh(
        results_df['feature'],
        results_df['cramers_v'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )

    ax1.set_xlabel("Cramer's V (сила связи)", fontsize=12)
    ax1.set_title(f'Влияние категориальных признаков на отток\n(просмотр > {threshold_minutes} мин, n={len(filtered_df):,})',
                 fontsize=14, fontweight='bold')

    # Добавляем линии порогов
    ax1.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Слабая связь (0.1)')
    ax1.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Сильная связь (0.3)')

    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        width = bar.get_width()
        p_val = results_df.iloc[i]['p_value']
        sig_text = '✓' if results_df.iloc[i]['significant'] else '✗'
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f} (p={p_val:.3f}) {sig_text}',
                va='center', fontsize=10, fontweight='bold')

    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # 2. Детальные графики по категориям для каждого признака
    num_features = len(detailed_analysis)
    if num_features > 0:
        cols = min(2, num_features)
        rows = (num_features + 1) // 2

        fig2, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        if num_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, analysis in enumerate(detailed_analysis[:len(axes)]):
            ax = axes[idx]
            feature_name = analysis['feature'].upper()
            churn_rates = analysis['churn_rates']
            category_counts = analysis['category_counts']

            # Сортируем по оттоку для лучшей визуализации
            sorted_idx = churn_rates.sort_values(ascending=False).index
            sorted_rates = churn_rates.loc[sorted_idx]
            sorted_counts = category_counts.loc[sorted_idx]

            # Создаем bar chart
            bars = ax.bar(range(len(sorted_rates)), sorted_rates.values,
                         color=plt.cm.RdYlGn_r(sorted_rates.values),
                         alpha=0.7,
                         edgecolor='black')

            ax.set_xlabel('Категории', fontsize=10)
            ax.set_ylabel('Доля оттока', fontsize=10)
            ax.set_title(f'{feature_name}\n(Categories: {len(sorted_rates)})',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(sorted_rates)))
            ax.set_xticklabels([str(cat)[:15] for cat in sorted_rates.index],
                              rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, min(1.0, sorted_rates.max() * 1.3))
            ax.grid(axis='y', alpha=0.3)

            # Добавляем значения на столбцы
            for i, (bar, rate, count) in enumerate(zip(bars, sorted_rates.values, sorted_counts.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.1%}\n({count})',
                       ha='center', va='bottom', fontsize=8)

            # Выделяем категорию с максимальным оттоком
            if len(sorted_rates) > 0:
                max_idx = sorted_rates.argmax()
                bars[max_idx].set_edgecolor('red')
                bars[max_idx].set_linewidth(2)

        # Скрываем пустые subplots
        for idx in range(len(detailed_analysis), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
    else:
        fig2 = None

    # Формируем текстовый отчет
    report = f"""
    ## Анализ категориальных признаков для активных пользователей

    ### Общая статистика:
    - **Порог просмотра:** > {threshold_minutes} минут
    - **Пользователей в анализе:** {len(filtered_df):,}
    - **Общий отток в группе:** {filtered_df['churn'].mean():.1%}
    - **Проанализировано признаков:** {len(results_df)}
    - **Статистически значимых (p < {p_value_threshold}):** {results_df['significant'].sum()}

    ### Рейтинг признаков по силе влияния (Cramer's V):
    """

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        sig_symbol = "✅" if row['significant'] else "❌"
        strength = "сильная" if row['cramers_v'] >= 0.3 else "средняя" if row['cramers_v'] >= 0.1 else "слабая"

        report += f"""
    {i}. **{row['feature'].upper()}** {sig_symbol}
       - Cramer's V = {row['cramers_v']:.3f} ({strength} связь)
       - p-value = {row['p_value']:.4f} {'(значимо)' if row['significant'] else '(не значимо)'}
       - Категорий: {row['n_categories']}
       - Макс. разница в оттоке: {row['churn_diff']:.1%}
       - Высокий отток: {row['highest_churn_category']} ({row['highest_churn_rate']:.1%})
       - Низкий отток: {row['lowest_churn_category']} ({row['lowest_churn_rate']:.1%})
    """

    report += f"""

    ### Заключение:
    {'Найдены статистически значимые факторы оттока среди активных пользователей. Рекомендуем сфокусироваться на категориях с наибольшим оттоком.'
     if results_df['significant'].any()
     else 'Не найдено статистически значимых факторов оттока среди активных пользователей. Возможно, отток равномерно распределен по всем категориям.'}

    ### Интерпретация метрик:
    - **Cramer's V < 0.1**: слабая связь
    - **0.1 ≤ Cramer's V < 0.3**: средняя связь
    - **Cramer's V ≥ 0.3**: сильная связь
    - **p-value < {p_value_threshold}**: связь статистически значима
    """

    return fig1, fig2, report

def create_categorical_interface(df):
    """
    Создает Gradio интерфейс для анализа категориальных признаков
    """
    # Определяем разумные пределы для слайдера
    max_time = int(df['avg_min_watch_daily'].max())
    median_time = int(df['avg_min_watch_daily'].median())

    # Проверяем наличие категориальных колонок
    cat_columns = ['city', 'device', 'source', 'favourite_genre']
    available_columns = [col for col in cat_columns if col in df.columns]

    print(f"Доступные колонки: {available_columns}")

    with gr.Blocks(title="Анализ категорий на отток", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Анализ влияния категорий на отток активных пользователей")
        gr.Markdown("Исследуем, как город, устройство, источник трафика и любимый жанр влияют на отток среди пользователей с высоким временем просмотра")

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

    return demo

def run_categorical_analysis_app(df):
    """
    Запускает приложение для анализа категориальных признаков
    """
    try:
        demo = create_categorical_interface(df)
        demo.launch(share=False, debug=False)
    except Exception as e:
        print(f"\nОшибка при запуске: {e}")





# Гипотеза 2
def analyze_threshold(threshold_minutes, df):
    """
    Анализирует порог времени просмотра для оттока пользователей
    """
    # Разделяем данные по порогу
    below_threshold = df[df['avg_min_watch_daily'] < threshold_minutes]
    above_threshold = df[df['avg_min_watch_daily'] >= threshold_minutes]

    # Проверяем, есть ли данные в обеих группах
    if len(below_threshold) == 0 or len(above_threshold) == 0:
        error_msg = f"Ошибка: Одна из групп пуста при пороге {threshold_minutes} минут"
        return None, None, error_msg, "0", "0", "0", "0"

    # Подсчитываем статистики
    counts = [below_threshold['churn'].sum(), above_threshold['churn'].sum()]
    nobs = [len(below_threshold), len(above_threshold)]

    # Статистический тест
    try:
        z_stat, p_value = proportions_ztest(counts, nobs, alternative='larger')
    except Exception as e:
        return None, None, f"Ошибка в статистическом тесте: {str(e)}", "0", "0", "0", "0"

    # Рассчитываем проценты оттока
    p1 = counts[0] / nobs[0]  # Доля оттока в группе < порога
    p2 = counts[1] / nobs[1]  # Доля оттока в группе ≥ порога

    # Относительный риск
    rr = p1 / p2 if p2 > 0 else float('inf')

    # Доверительные интервалы (95%)
    ci_below = proportion_confint(count=counts[0], nobs=nobs[0], alpha=0.05)
    ci_above = proportion_confint(count=counts[1], nobs=nobs[1], alpha=0.05)

    # Создаем первый график - Процент оттока
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Данные для графика
    percentages = [p1 * 100, p2 * 100]
    group_names = [f'< {threshold_minutes} мин', f'≥ {threshold_minutes} мин']
    colors_churn = ['#FF5252', '#2ECC71']

    # Создаем столбчатую диаграмму
    bars = ax1.bar([0, 1], percentages, color=colors_churn, alpha=0.8,
                   edgecolor='black', width=0.6)

    ax1.set_title(f'Процент оттока по группам (порог: {threshold_minutes} минут)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Процент оттока (%)', fontsize=12)
    ax1.set_xlabel('Группы', fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(group_names, fontsize=12)
    ax1.set_ylim(0, max(percentages) * 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # Добавляем подписи над столбиками
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Процент
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        # Абсолютные числа
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{counts[i]:,} из {nobs[i]:,}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # Линия сравнения и разница
    diff = p1 - p2
    ax1.plot([0, 1], [percentages[0], percentages[1]], 'k--',
             alpha=0.5, linewidth=1)
    ax1.text(0.5, max(percentages) * 1.05,
            f'Разница: {diff*100:.1f}%\nОтносительный риск: {rr:.1f}×',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Информация о статистической значимости
    if p_value < 0.05:
        significance = 'Статистически значимо (p < 0.05)'
        sig_color = 'lightgreen'
    else:
        significance = 'Не статистически значимо (p ≥ 0.05)'
        sig_color = 'lightcoral'

    ax1.text(0.5, -max(percentages) * 0.15,
            f'{significance} | Z = {z_stat:.2f} | p-value = {p_value:.6f}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=sig_color, alpha=0.8))

    plt.tight_layout()

    # Создаем второй график - Доверительные интервалы
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Отображаем доверительные интервалы
    ax2.errorbar(x=0, y=p1,
                yerr=[[p1 - ci_below[0]], [ci_below[1] - p1]],
                fmt='o', capsize=10, capthick=2, markersize=10,
                color='red', ecolor='black', linewidth=2,
                label=f'< {threshold_minutes} мин')

    ax2.errorbar(x=1, y=p2,
                yerr=[[p2 - ci_above[0]], [ci_above[1] - p2]],
                fmt='o', capsize=10, capthick=2, markersize=10,
                color='green', ecolor='black', linewidth=2,
                label=f'≥ {threshold_minutes} мин')

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'< {threshold_minutes} мин', f'≥ {threshold_minutes} мин'],
                       fontsize=12)
    ax2.set_ylabel('Доля оттока', fontsize=12)
    ax2.set_title('Доверительные интервалы (95%) для долей оттока',
                  fontsize=14, fontweight='bold')

    # Добавляем значения процентов
    ax2.text(0, p1 + 0.02, f'{p1:.1%}', ha='center',
            fontsize=11, fontweight='bold')
    ax2.text(1, p2 + 0.02, f'{p2:.1%}', ha='center',
            fontsize=11, fontweight='bold')

    # Проверяем пересечение доверительных интервалов
    if ci_below[1] < ci_above[0] or ci_above[1] < ci_below[0]:
        ci_significance = "Доверительные интервалы НЕ пересекаются\n(статистически значимо)"
        ci_color = 'lightgreen'
    else:
        ci_significance = "Доверительные интервалы пересекаются\n(не статистически значимо)"
        ci_color = 'lightcoral'

    ax2.text(0.5, max(p1, p2) + 0.1, ci_significance,
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=ci_color, alpha=0.8))

    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=11)
    plt.tight_layout()

    # Формируем текстовый вывод
    stats_text = f"""
    ## Статистические результаты для порога {threshold_minutes} минут:

    ### Статистика групп:
    - **Группа < {threshold_minutes} мин:** {nobs[0]:,} пользователей, {counts[0]:,} отток ({p1:.1%})
    - **Группа ≥ {threshold_minutes} мин:** {nobs[1]:,} пользователей, {counts[1]:,} отток ({p2:.1%})

    ### Статистические тесты:
    - **Z-статистика:** {z_stat:.2f}
    - **p-value:** {p_value:.6f}

    ### Вывод:
    {significance}. {'Рекомендуем использовать этот порог для прогнозирования оттока.' if p_value < 0.05 else 'Порог не является статистически значимым для прогнозирования оттока.'}
    """

    return fig1, fig2, stats_text, str(z_stat), str(p_value), str(rr), significance

# Главная функция запуска
def launch_gradio_interface(df, initial_threshold=30):
    """
    Запускает Gradio интерфейс в Colab
    """
    print(f"Загружено {len(df):,} записей")

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

    # Создаем интерфейс
    with gr.Blocks(title="Анализ порога времени просмотра", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Анализ влияния времени просмотра на отток пользователей")
        gr.Markdown(f"**Всего данных:** {len(df):,} пользователей | **Общий отток:** {df['churn'].mean():.1%}")

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
            outputs=[plot1, plot2, results_text, z_stat_output, p_value_output, rr_output, significance_output]
        )

        # Инициализируем с начальным значением
        demo.load(
            fn=lambda: update_wrapper(initial_threshold),
            outputs=[plot1, plot2, results_text, z_stat_output, p_value_output, rr_output, significance_output]
        )

        gr.Markdown("---")
        gr.Markdown("### ℹКак интерпретировать результаты:")
        gr.Markdown("""
        - **p-value < 0.05**: разница статистически значима
        - **Относительный риск > 1**: меньший просмотр = больший риск оттока
        - **Доверительные интервалы не пересекаются**: подтверждение значимости
        """)

    # Запускаем в Colab-совместимом режиме
    try:
        demo.launch(debug=False, share=False)
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        demo.launch(share=True, debug=False)

def run_gradio_app(df, initial_threshold=30):
    launch_gradio_interface(df, initial_threshold)
