# Опись репозитория exam_utils

## Владелец: [KateChirkova](https://github.com/KateChirkova)
## Репозиторий: https://github.com/KateChirkova/exam_utils

## Назначение
Вспомогательные Python-утилиты для **практических** заданий на экзамене.
Заменяют рутину — **не решают задачу целиком**.

Ориентир: [exam-preparation-utils](https://github.com/Moldarus/exam-preparation-utils).

---

## Практические задачи → какие утилиты использовать

### Задача 1. RAG-сервис (XML Wiki, векторная БД, `/llm?query=`)
**Ваш код:** FastAPI/Flask, маршруты, системный промпт, гибридный поиск.  
**Утилиты `rag_helpers.py`:**

| Функция | Рутина |
|---------|--------|
| `load_xml_documents(source_dir)` | чтение XML из `./source_data/` |
| `chunk_documents(docs)` | разбиение на чанки для embedding |
| `load_embedding_model(name)` | загрузка multilingual embedding (русский) |
| `embed_texts(model, texts)` | векторизация |
| `init_chroma_store(persist_dir)` | создание векторной БД ChromaDB |
| `index_chunks(collection, chunks, model)` | запись векторов в БД |
| `search_chunks(collection, query, model, top_k)` | поиск по запросу |
| `join_retrieved_context(hits)` | склейка найденных чанков |
| `format_llm_json_response(answer)` | `{"answer": ...}` для страницы |
| `startup_status_message/html(loaded)` | текст/HTML стартовой страницы |
| `write_daily_log(msg, log_dir, error)` | `logs/log-ДАТА.txt` |
| `safe_index_on_startup(...)` | индексация с перехватом ошибок |

### Задача 2. Тональность + NER + отчёт с таблицами и графиками
| Функция | Рутина |
|---------|--------|
| `read_text_file`, `split_text_to_paragraphs` | чтение и разбиение текста |
| `load_sentiment_pipeline`, `load_ner_pipeline` | загрузка моделей (рус.) |
| `analyze_sentiment_batch`, `analyze_ner_batch` | пакетный анализ → DataFrame |
| `plot_sentiment_chart`, `plot_ner_top_entities` | графики |
| `save_nlp_analysis_report(...)` | CSV + PNG в папку отчёта |

### Задача 3. Прогноз реакций на публикацию
| Функция | Рутина |
|---------|--------|
| `extract_social_text_features(text)` | длина, хэштеги, emoji и т.д. |
| `social_features_dataframe(texts)` | признаки для всех текстов |
| `merge_features_with_target(...)` | склейка с целевой переменной |
| + `utils.prepare_xy`, `train_regression_models` | обучение — в вашем коде |

### Задача 4. Векторизация (TF-IDF, Word2Vec, Skip-Gram, pretrained)
| Функция | Рутина |
|---------|--------|
| `vectorize_tfidf`, `vectorize_count` | sklearn-векторизаторы |
| `train_word2vec(sg=1)` | Skip-Gram; `sg=0` — CBOW |
| `document_vector_from_word2vec` | вектор документа |
| `pretrained_embeddings` | sentence-transformers |
| `vectorization_summary_table` | сравнение размерностей |
| `reduce_embeddings_2d`, `plot_embeddings_2d` | визуализация |

### Задача 5. Диалоговая система, сравнение LLM
| Функция | Рутина |
|---------|--------|
| `llm_helpers.chat(messages)` | вызов LLM (промпт — ваш) |
| `compare_llm_on_questions(questions, providers)` | одни вопросы → разные модели |
| `save_model_comparison_report(df)` | Markdown-отчёт |
| `preview_comparison_table(df)` | preview в консоли |

---

## Остальные модули

### `utils.py` — EDA, визуализация, ML-пайплайн (общая рутина)
`quick_eda`, `quick_visualization_report`, `prepare_xy`, `create_ml_pipeline`, и др.

### `llm_helpers.py` — только подключение
`load_llm_config`, `get_llm_client`, `chat`, `chat_stream`, `check_connection`

### `exam_examples.py` — примеры по всем блокам

## Переменные окружения (`.env`)
```
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
# GigaChat и др. — через OPENAI_BASE_URL + ключ
# Ollama: OLLAMA_BASE_URL=http://localhost:11434
```

## Установка
```bash
git clone https://github.com/KateChirkova/exam_utils.git
cd exam_utils
python -m pip install -r requirements.txt
python exam_examples.py
```
