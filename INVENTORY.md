# Опись репозитория для государственного экзамена / ГЭК

## Владелец: [KateChirkova](https://github.com/KateChirkova)
## Репозиторий: `https://github.com/KateChirkova/om-gek-exam-utils`

## Назначение
Набор Python-утилит для **ускорения** решения практических задач из файла `ОМ_ГЭК.docx`.
Утилиты **не содержат готовых решений** экзаменационных задач на Java.

## Источник задач
- Практика Java: CompletableFuture, LRU Cache, REST «Библиотека», парсинг `server.log`
- Теория: SQL, алгоритмы, Linux, сети, архитектура, Docker, Git, ML, распределённые системы, ITIL, Cloud

## Структура репозитория

### 1. `utils.py` (основной модуль)
| Блок | Функции | Назначение |
|------|---------|------------|
| Параллелизм | `generate_random_int_list`, `split_into_chunks`, `reference_sum_of_squares`, `benchmark_call`, `plot_parallel_chunks` | Тестовые данные и сверка результата Java |
| LRU | `generate_lru_test_sequence`, `print_lru_trace_table`, `plot_lru_access_pattern` | Трассировка get/put, не реализация кэша |
| REST | `generate_sample_books`, `validate_book_dto`, `books_to_json`, `print_curl_templates`, `print_rest_layer_skeleton` | DTO, валидация, HTTP-шаблоны |
| Logs | `get_log_line_pattern`, `parse_log_line`, `generate_sample_server_log`, `format_log_report`, `plot_log_errors_by_service` | Regex, генерация log, формат вывода |
| Теория | `print_sql_cheatsheet`, `print_git_cheatsheet`, `print_network_reference`, `compute_classification_metrics` | Шпаргалки и быстрые проверки |
| LLM | см. `llm_helpers.py` | Подключение к OpenAI / Ollama |

### 2. `llm_helpers.py`
- `load_llm_config`, `get_llm_client`, `ask_llm`
- `explain_concept`, `review_code_snippet` — с ограничением «не давать готовое решение»

### 3. `exam_examples.py`
Демонстрация всех блоков утилит.

### 4. `data/sample_server.log`
Пример лог-файла для локальных тестов парсера.

### 5. `requirements.txt`
Зависимости с открытыми лицenzиями.

## Переменные окружения (`.env`)
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# опционально для Ollama:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Использование на экзамене
1. Клонировать репозиторий на рабочую машину
2. `pip install -r requirements.txt`
3. Импортировать нужные функции в интерактивной сессии или скрипте проверки
4. Решение задач писать на Java (как требует билет)

## Ориентир
Структура и подход заимствованы из [exam-preparation-utils](https://github.com/Moldarus/exam-preparation-utils).
