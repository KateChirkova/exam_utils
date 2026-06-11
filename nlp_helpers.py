"""
Вспомогательные функции для NLP/LLM практических задач (2–5).
Не решают задачи целиком — ускоряют чтение текста, анализ, векторизацию, отчёты.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils import plot_bar_counts, save_figure, setup_plot_style


# =============================================================================
# Общее: чтение текста
# =============================================================================

def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Прочитать текстовый файл целиком."""
    return Path(path).read_text(encoding=encoding)


def split_text_to_paragraphs(text: str, min_len: int = 20) -> list[str]:
    """Разбить текст на абзацы (для пакетного анализа)."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def texts_to_dataframe(texts: Sequence[str], label: str = "text") -> pd.DataFrame:
    """Список текстов → DataFrame с id."""
    return pd.DataFrame({"id": range(len(texts)), label: list(texts)})


# =============================================================================
# Задача 2: тональность + NER + отчёт
# =============================================================================

def load_sentiment_pipeline(model: str = "blanchefort/rubert-base-cased-sentiment"):
    """Загрузить pipeline тональности (русский). Модель выбираете под задачу."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc
    return pipeline("sentiment-analysis", model=model)


def load_ner_pipeline(model: str = "Gherman/bert-base-NER-Russian"):
    """Загрузить pipeline NER (русский)."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc
    return pipeline("ner", model=model, aggregation_strategy="simple")


def analyze_sentiment_batch(pipe, texts: Sequence[str]) -> pd.DataFrame:
    """Тональность для списка текстов → таблица."""
    rows = []
    for i, text in enumerate(texts):
        result = pipe(text[:512])[0]
        rows.append({
            "id": i,
            "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
            "label": result.get("label"),
            "score": round(float(result.get("score", 0)), 4),
        })
    return pd.DataFrame(rows)


def analyze_ner_batch(pipe, texts: Sequence[str]) -> pd.DataFrame:
    """NER для списка текстов → таблица сущностей."""
    rows = []
    for i, text in enumerate(texts):
        for ent in pipe(text[:512]):
            rows.append({
                "text_id": i,
                "entity": ent.get("word"),
                "type": ent.get("entity_group") or ent.get("entity"),
                "score": round(float(ent.get("score", 0)), 4),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["text_id", "entity", "type", "score"])


def plot_sentiment_chart(sentiment_df: pd.DataFrame, title: str = "Sentiment distribution") -> None:
    """График распределения тональности."""
    counts = sentiment_df["label"].value_counts()
    plot_bar_counts(counts, title=title)


def plot_ner_top_entities(ner_df: pd.DataFrame, top_n: int = 15, title: str = "Top entities") -> None:
    """Bar chart топ сущностей."""
    if ner_df.empty:
        print("No entities to plot")
        return
    counts = ner_df["entity"].value_counts().head(top_n)
    plot_bar_counts(counts, title=title, horizontal=True)


def save_nlp_analysis_report(
    sentiment_df: pd.DataFrame,
    ner_df: pd.DataFrame,
    output_dir: str | Path = "./reports",
    prefix: str = "nlp_report",
) -> dict[str, Path]:
    """
    Сохранить таблицы (CSV) + графики (PNG) — каркас отчёта для задачи 2.
    Текстовые выводы и шаблон отчёта дописываете сами.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    sent_csv = output_dir / f"{prefix}_sentiment.csv"
    sentiment_df.to_csv(sent_csv, index=False, encoding="utf-8-sig")
    paths["sentiment_csv"] = sent_csv

    ner_csv = output_dir / f"{prefix}_ner.csv"
    ner_df.to_csv(ner_csv, index=False, encoding="utf-8-sig")
    paths["ner_csv"] = ner_csv

    setup_plot_style()
    if not sentiment_df.empty:
        plot_sentiment_chart(sentiment_df)
        paths["sentiment_png"] = save_figure(output_dir / f"{prefix}_sentiment.png")

    if not ner_df.empty:
        plot_ner_top_entities(ner_df)
        paths["ner_png"] = save_figure(output_dir / f"{prefix}_ner.png")

    print(f"Report artifacts saved to {output_dir}")
    return paths


# =============================================================================
# Задача 3: признаки текста для прогноза реакций
# =============================================================================

def extract_social_text_features(text: str) -> dict:
    """Базовые признаки текста публикации (рутина feature engineering)."""
    words = text.split()
    return {
        "char_len": len(text),
        "word_count": len(words),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "hashtag_count": len(re.findall(r"#\w+", text)),
        "mention_count": len(re.findall(r"@\w+", text)),
        "emoji_count": len(re.findall(r"[\U0001F300-\U0001FAFF]", text)),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }


def social_features_dataframe(texts: Sequence[str]) -> pd.DataFrame:
    """Признаки для набора текстов → DataFrame (дальше — ваша модель регрессии)."""
    rows = [extract_social_text_features(t) for t in texts]
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(texts)))
    return df


def merge_features_with_target(features_df: pd.DataFrame, targets: Sequence, target_col: str = "reactions") -> pd.DataFrame:
    """Склеить признаки с целевой переменной."""
    out = features_df.copy()
    out[target_col] = list(targets)
    return out


# =============================================================================
# Задача 4: векторизация (TF-IDF, Count, Word2Vec, предобученные embeddings)
# =============================================================================

def vectorize_tfidf(texts: Sequence[str], max_features: int = 5000) -> tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    print(f"TF-IDF: shape={matrix.shape}")
    return matrix.toarray(), vec


def vectorize_count(texts: Sequence[str], max_features: int = 5000) -> tuple[np.ndarray, CountVectorizer]:
    vec = CountVectorizer(max_features=max_features)
    matrix = vec.fit_transform(texts)
    print(f"Count: shape={matrix.shape}")
    return matrix.toarray(), vec


def train_word2vec(
    texts: Sequence[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 10,
):
    """
    Обучить Word2Vec. sg=1 → Skip-Gram, sg=0 → CBOW.
    Возвращает модель gensim.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError("pip install gensim") from exc

    tokenized = [t.lower().split() for t in texts]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
    )
    mode = "Skip-Gram" if sg == 1 else "CBOW"
    print(f"Word2Vec ({mode}): vocab={len(model.wv)}")
    return model


def document_vector_from_word2vec(model, text: str) -> np.ndarray:
    """Вектор документа — среднее векторов слов (для сравнения методов)."""
    words = [w for w in text.lower().split() if w in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in words], axis=0)


def pretrained_embeddings(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """Embeddings предобученной моделью (русский текст)."""
    from rag_helpers import load_embedding_model, embed_texts

    model = load_embedding_model(model_name)
    return np.array(embed_texts(model, texts))


def vectorization_summary_table(methods: dict[str, np.ndarray]) -> pd.DataFrame:
    """Сводная таблица размерностей векторов по методам."""
    rows = [{"method": name, "shape": str(arr.shape), "dims": arr.shape[-1]} for name, arr in methods.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def reduce_embeddings_2d(
    vectors: np.ndarray,
    method: str = "tsne",
    random_state: int = 42,
) -> np.ndarray:
    """Снизить размерность для визуализации (PCA / t-SNE)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(vectors) - 1))

    return reducer.fit_transform(vectors)


def plot_embeddings_2d(
    coords: np.ndarray,
    labels: Sequence | None = None,
    title: str = "Embeddings 2D",
) -> None:
    """Scatter 2D-проекции векторов."""
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Задача 5: сравнение ответов разных LLM
# =============================================================================

def compare_llm_on_questions(
    questions: Sequence[str],
    providers: Sequence[str] = ("openai", "ollama"),
    message_builder: Callable[[str], list[dict[str, str]]] | None = None,
) -> pd.DataFrame:
    """
    Один и тот же список вопросов → разные LLM → таблица для отчёта.
    message_builder(question) → list[dict] — формат сообщений задаёте сами.
    """
    from llm_helpers import chat

    if message_builder is None:
        message_builder = lambda q: [{"role": "user", "content": q}]

    rows = []
    for q in questions:
        for provider in providers:
            try:
                answer = chat(message_builder(q), provider=provider)
                status = "ok"
            except Exception as exc:
                answer = str(exc)
                status = "error"
            rows.append({"question": q, "provider": provider, "status": status, "answer": answer})
    df = pd.DataFrame(rows)
    return df


def save_model_comparison_report(
    comparison_df: pd.DataFrame,
    output_path: str | Path = "./reports/llm_comparison.md",
) -> Path:
    """Сохранить сравнение моделей в Markdown (каркас отчёта)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Сравнение ответов LLM\n"]
    for question in comparison_df["question"].unique():
        lines.append(f"## Вопрос: {question}\n")
        subset = comparison_df[comparison_df["question"] == question]
        for _, row in subset.iterrows():
            lines.append(f"### {row['provider']} ({row['status']})\n")
            lines.append(f"{row['answer']}\n")
        lines.append("---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison report: {output_path}")
    return output_path


def preview_comparison_table(comparison_df: pd.DataFrame, max_answer_len: int = 120) -> pd.DataFrame:
    """Краткий preview для консоли."""
    preview = comparison_df.copy()
    preview["answer"] = preview["answer"].str[:max_answer_len] + "..."
    print(preview.to_string(index=False))
    return preview
