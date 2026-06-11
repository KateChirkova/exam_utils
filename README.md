```bash
git clone https://github.com/KateChirkova/exam_utils.git
cd exam_utils
python -m pip install -r requirements.txt
```

| Функция | Назначение |
|---------|------------|
| `setup_plot_style` | единый стиль графиков |
| `save_figure` | сохранить фигуру в файл |
| `load_table` | загрузить CSV / Excel / JSON |
| `preview_df` | shape, dtypes, head |
| `plot_bar_counts` | bar chart |
| `plot_distributions` | гистограммы числовых колонок |
| `plot_correlation_matrix` | тепловая карта корреляций |
| `plot_clusters_2d` | scatter кластеров в 2D |
| `plot_confusion_matrix_custom` | матрица ошибок |
| `plot_roc_curve` | ROC-кривая и AUC |
| `plot_predictions` | факт vs прогноз |
| `plot_parallel_coordinates` | параллельные координаты |
| `quick_visualization_report` | корреляции, распределения и целевая колонка |
| `load_llm_config` | настройки LLM из `.env` |
| `check_llm_ready` | проверка настроек LLM |

Модуль `utils.py` — первые 13 функций. Модуль `llm_helpers.py` — последние 2. Вызов LLM (`ollama`, LM Studio и др.) — в вашем коде.
