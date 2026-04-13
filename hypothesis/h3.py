import matplotlib.pyplot as plt
import numpy as np


def create_engagement_analysis(df, threshold_minutes, min_days_active, comparison_days):
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