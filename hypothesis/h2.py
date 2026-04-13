import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

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