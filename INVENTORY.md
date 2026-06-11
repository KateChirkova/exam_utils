# Опись репозитория exam_utils

## Владелец: [KateChirkova](https://github.com/KateChirkova)
## Репозиторий: https://github.com/KateChirkova/exam_utils

## Назначение
Python-утилиты для **практических** заданий на экзамене. Заменяют рутину:
загрузка данных, EDA, графики, предобработка, обучение моделей, парсинг файлов.
**Не решают конкретную задачу целиком.**

Ориентир по структуре: [exam-preparation-utils](https://github.com/Moldarus/exam-preparation-utils).

## Файлы

### `utils.py`
| Блок | Функции |
|------|---------|
| Рутина | `setup_plot_style`, `save_figure`, `preview_df`, `load_table`, `read_text_lines`, `timer`, `benchmark_call`, `split_into_chunks`, `group_and_count`, `top_n_counts`, `extract_with_regex` |
| EDA | `quick_eda` |
| Визуализация | `plot_distributions`, `plot_correlation_matrix`, `plot_bar_counts`, `plot_box_by_category`, `plot_scatter_colored`, `quick_visualization_report`, `plot_parallel_coordinates` |
| Предобработка | `handle_missing_values`, `encode_categorical`, `scale_features`, `prepare_xy` |
| Кластеризация | `kmeans_elbow_method`, `perform_clustering`, `plot_clusters_2d` |
| Классификация | `train_classification_models`, `plot_confusion_matrix`, `plot_roc_curve` |
| Регрессия | `train_regression_models`, `plot_predictions` |
| Аномалии | `detect_anomalies_iqr`, `detect_anomalies_zscore` |
| Features | `create_interaction_features`, `select_features_by_correlation`, `cross_validate_model` |
| Данные / ML | `generate_sample_data`, `create_ml_pipeline` |

### `exam_examples.py`
8 примеров использования всех блоков.

### `llm_helpers.py`
Только подключение — без готовых промптов:
`load_llm_config`, `get_llm_client`, `chat`, `chat_stream`, `check_connection`.

### `requirements.txt`
Зависимости (numpy, pandas, sklearn, matplotlib, seaborn, openai).

## Использование
```bash
git clone https://github.com/KateChirkova/exam_utils.git
cd exam_utils
python -m pip install -r requirements.txt
python exam_examples.py
```
