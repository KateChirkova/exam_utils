# Опись репозитория exam_utils

## Владелец: [KateChirkova](https://github.com/KateChirkova)
## Репозиторий: https://github.com/KateChirkova/exam_utils

## Назначение
Сокращение рутины на экзамене: **графики, загрузка данных, подключение LLM**.
Без обучения моделей, предобработки и готовых промптов.

## `utils.py` — только визуализация

| Функция | Назначение |
|---------|------------|
| `setup_plot_style` | единый стиль графиков |
| `save_figure` | сохранить fig в файл |
| `load_table` | csv / xlsx / json |
| `preview_df` | shape + head |
| `plot_bar_counts` | bar chart |
| `plot_distributions` | гистограммы |
| `plot_correlation_matrix` | тепловая карта корреляций |
| `plot_clusters_2d` | scatter кластеров |
| `plot_confusion_matrix_custom` | **матрица ошибок** |
| `plot_roc_curve` | ROC + AUC |
| `plot_predictions` | факт vs прогноз |
| `plot_parallel_coordinates` | параллельные координаты |
| `quick_visualization_report` | корреляции + распределения + целевая |

## `llm_helpers.py` — только подключение

| Функция | Назначение |
|---------|------------|
| `load_llm_config` | настройки из `.env` |
| `get_llm_client` | OpenAI-совместимый клиент |

## Установка
```bash
git clone https://github.com/KateChirkova/exam_utils.git
cd exam_utils
python -m pip install -r requirements.txt
```
