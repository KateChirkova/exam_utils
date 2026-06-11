"""
Вспомогательные функции для задачи RAG-сервиса (задача 1).
Не реализует API-сервис — только рутину: XML, чанки, векторная БД, логи, HTML-статусы.
"""

from __future__ import annotations

import json
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

# Модель по умолчанию — мультиязычная, подходит для русского Wiki-XML
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_xml_documents(source_dir: str | Path = "./source_data") -> list[dict]:
    """
    Прочитать все XML из папки → список {source, title, text}.
    Пустая папка / нет файлов → ValueError (обработайте в своём сервисе).
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    xml_files = sorted(source_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files in {source_dir}")

    documents: list[dict] = []
    for path in xml_files:
        tree = ET.parse(path)
        root = tree.getroot()
        articles = root.findall(".//article")
        if articles:
            for i, art in enumerate(articles):
                title = _xml_find_text(art, "title") or f"{path.stem}_{i}"
                body = _xml_collect_text(art)
                if body.strip():
                    documents.append({"source": path.name, "title": title, "text": body.strip()})
        else:
            title = _xml_find_text(root, "title") or path.stem
            body = _xml_collect_text(root)
            if body.strip():
                documents.append({"source": path.name, "title": title, "text": body.strip()})

    print(f"Loaded {len(documents)} documents from {len(xml_files)} XML files")
    return documents


def _xml_find_text(node: ET.Element, tag: str) -> str | None:
    el = node.find(f".//{tag}")
    return el.text.strip() if el is not None and el.text else None


def _xml_collect_text(node: ET.Element) -> str:
    parts = [t.strip() for t in node.itertext() if t and t.strip()]
    return " ".join(parts)


def chunk_documents(
    documents: Sequence[dict],
    chunk_size: int = 500,
    overlap: int = 50,
    text_key: str = "text",
) -> list[dict]:
    """Разбить документы на чанки для embedding (скользящее окно по словам)."""
    chunks: list[dict] = []
    for doc in documents:
        words = doc[text_key].split()
        if not words:
            continue
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + chunk_size])
            if piece:
                chunks.append({**doc, "chunk": piece, "chunk_id": len(chunks)})
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Загрузить SentenceTransformer (русский/мультиязычный embedding)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("pip install sentence-transformers") from exc
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(model, texts: Sequence[str]) -> list[list[float]]:
    """Векторизовать список текстов."""
    vectors = model.encode(list(texts), show_progress_bar=len(texts) > 20)
    return vectors.tolist()


def init_chroma_store(persist_dir: str | Path = "./chroma_db", collection: str = "wiki"):
    """Создать / подключить ChromaDB collection."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("pip install chromadb") from exc

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
    print(f"Chroma collection '{collection}' at {persist_dir}")
    return col


def index_chunks(
    collection,
    chunks: Sequence[dict],
    embed_model,
    text_key: str = "chunk",
    batch_size: int = 64,
) -> int:
    """Добавить чанки в векторную БД. Возвращает число проиндексированных."""
    texts = [c[text_key] for c in chunks]
    ids = [str(c.get("chunk_id", i)) for i, c in enumerate(chunks)]
    metadatas = [{"source": c.get("source", ""), "title": c.get("title", "")} for c in chunks]

    total = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_meta = metadatas[start : start + batch_size]
        vectors = embed_texts(embed_model, batch_texts)
        collection.add(ids=batch_ids, documents=batch_texts, embeddings=vectors, metadatas=batch_meta)
        total += len(batch_texts)

    print(f"Indexed {total} chunks")
    return total


def search_chunks(
    collection,
    query: str,
    embed_model,
    top_k: int = 5,
) -> list[dict]:
    """Поиск по векторной БД — вернуть найденные чанки с distance."""
    query_vec = embed_texts(embed_model, [query])[0]
    result = collection.query(query_embeddings=[query_vec], n_results=top_k)
    hits = []
    for i in range(len(result["ids"][0])):
        hits.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i] if result.get("distances") else None,
        })
    return hits


def join_retrieved_context(hits: Sequence[dict], separator: str = "\n---\n") -> str:
    """Склеить найденные чанки в один контекст (для передачи в LLM — логику промпта пишете сами)."""
    return separator.join(h["text"] for h in hits if h.get("text"))


def format_llm_json_response(answer: str) -> str:
    """Формат ответа {"answer": ...} для страницы /llm."""
    return json.dumps({"answer": answer}, ensure_ascii=False, indent=2)


def startup_status_message(data_loaded: bool, start_time: datetime | None = None) -> str:
    """Текст статуса при запуске (как в ТЗ)."""
    ts = (start_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    if data_loaded:
        return f"Сервер запущен. Дата и Время запуска: {ts}"
    return "Сервер запущен. Локальные данные не загружены"


def startup_status_html(data_loaded: bool, start_time: datetime | None = None) -> str:
    """HTML стартовой страницы."""
    msg = startup_status_message(data_loaded, start_time)
    return f"<!DOCTYPE html><html><body><h2>{msg}</h2></body></html>"


def write_daily_log(
    message: str,
    log_dir: str | Path = "./logs",
    error: BaseException | None = None,
) -> Path:
    """
    Запись в log-ДАТА.txt (один файл на день), как в ТЗ.
    Пример: logs/log-2026-06-11.txt
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = log_dir / f"log-{date_str}.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{ts}] {message}"]
    if error:
        lines.append(traceback.format_exc())
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def safe_index_on_startup(
    source_dir: str | Path,
    persist_dir: str | Path,
    embed_model_name: str = DEFAULT_EMBEDDING_MODEL,
    log_dir: str | Path = "./logs",
) -> tuple[bool, object | None, object | None]:
    """
    Обёртка «загрузить XML → проиндексировать» с перехватом ошибок.
    Возвращает (успех, collection, embed_model).
    Ошибки пишет в дневной лог — удобно для блока startup в вашем сервисе.
    """
    try:
        docs = load_xml_documents(source_dir)
        chunks = chunk_documents(docs)
        model = load_embedding_model(embed_model_name)
        col = init_chroma_store(persist_dir)
        index_chunks(col, chunks, model)
        return True, col, model
    except Exception as exc:
        write_daily_log("Vector DB indexing failed", log_dir, error=exc)
        return False, None, None
