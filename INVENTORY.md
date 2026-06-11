# Опись репозитория exam_utils

## Владелец: [KateChirkova](https://github.com/KateChirkova)
## Репозиторий: https://github.com/KateChirkova/exam_utils

## Назначение
Вспомогательные функции для экзамена — **сокращение рутины** (EDA, графики, предобработка, ML-отчёты).
По структуре ориентир: [exam-preparation-utils](https://github.com/Moldarus/exam-preparation-utils).

Не заменяет sklearn/pandas/transformers. Не решает задачу целиком.

## Файлы

### `utils.py`
| Блок | Функции |
|------|---------|
| Визуализация (коротко) | `setup_plot_style`, `save_figure`, `plot_bar_counts`, `load_table`, `preview_df` |
| EDA | `quick_eda` |
| Графики | `plot_distributions`, `plot_correlation_matrix`, `quick_visualization_report`, `plot_parallel_coordinates` |
| Предобработка | `handle_missing_values`, `encode_categorical`, `scale_features` |
| Кластеризация | `kmeans_elbow_method`, `perform_clustering`, `plot_clusters_2d` |
| Классификация | `train_classification_models`, `plot_confusion_matrix_custom`, `plot_roc_curve` |
| Регрессия | `train_regression_models`, `plot_predictions` |
| Аномалии | `detect_anomalies_iqr`, `detect_anomalies_zscore` |
| Features | `create_interaction_features`, `create_polynomial_features`, `select_features_by_variance`, `select_features_by_correlation`, `cross_validate_model` |
| Прочее | `generate_sample_data`, `create_ml_pipeline`, `demo_sql_queries` |

### `llm_helpers.py`
Подключение к LLM (без готовых промптов): `load_llm_config`, `get_llm_client`, `chat`, `chat_stream`, `check_connection`.

### `exam_examples.py` — примеры вызовов

## Установка
```bash
git clone https://github.com/KateChirkova/exam_utils.git
cd exam_utils
python -m pip install -r requirements.txt
```
