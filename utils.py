"""
Утилиты для подготовки к ГЭК / вступительному тесту (ОМ, IT-сервисы).
Направление: практические задания по Java + теоретический блок.

Важно: модуль НЕ решает экзаменационные задачи целиком.
Только генерация тестовых данных, проверка форматов, визуализация, шаблоны.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Параллельная обработка (задача: CompletableFuture / потоки)
# ---------------------------------------------------------------------------

def generate_random_int_list(
    n: int = 1_000_000,
    low: int = -1000,
    high: int = 1000,
    seed: int | None = 42,
) -> list[int]:
    """Сгенерировать список случайных int для тестирования Java-программы."""
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n)]


def split_into_chunks(data: Sequence[int], n_chunks: int) -> list[list[int]]:
    """Разбить последовательность на n_chunks частей (проверка границ чанков)."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    size = len(data)
    base, extra = divmod(size, n_chunks)
    chunks: list[list[int]] = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < extra else 0)
        chunks.append(list(data[start:end]))
        start = end
    return chunks


def reference_sum_of_squares(numbers: Iterable[int]) -> int:
    """Эталонная сумма квадратов — для сверки результата вашей Java-программы."""
    return sum(x * x for x in numbers)


def benchmark_call(label: str, func: Callable[[], Any], repeat: int = 1) -> dict[str, Any]:
    """Замер времени выполнения callable (мс). Не заменяет реализацию в Java."""
    times_ms: list[float] = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times_ms.append((time.perf_counter() - t0) * 1000)
    stats = {
        "label": label,
        "result": result,
        "times_ms": times_ms,
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }
    print(f"[{label}] mean={stats['mean_ms']:.2f} ms, min={stats['min_ms']:.2f}, max={stats['max_ms']:.2f}")
    return stats


def plot_parallel_chunks(
    chunk_sizes: Sequence[int],
    chunk_sums: Sequence[int] | None = None,
    title: str = "Parallel chunk processing",
) -> None:
    """Визуализация размеров чанков и (опционально) частичных сумм."""
    fig, axes = plt.subplots(1, 2 if chunk_sums else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].bar(range(len(chunk_sizes)), chunk_sizes, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Chunk index")
    axes[0].set_ylabel("Size")
    axes[0].set_title("Chunk sizes")

    if chunk_sums is not None:
        axes[1].bar(range(len(chunk_sums)), chunk_sums, color="coral", edgecolor="black")
        axes[1].set_xlabel("Chunk index")
        axes[1].set_ylabel("Sum of squares")
        axes[1].set_title("Partial sums (reference)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. LRU Cache — только трассировка и тестовые последовательности
# ---------------------------------------------------------------------------

@dataclass
class LRUOperation:
    op: str  # "get" | "put"
    key: Any
    value: Any | None = None


def generate_lru_test_sequence(
    capacity: int,
    n_ops: int = 20,
    key_pool: int = 8,
    seed: int = 42,
) -> list[LRUOperation]:
    """Случайная последовательность get/put для ручной или unit-проверки LRU."""
    rng = random.Random(seed)
    ops: list[LRUOperation] = []
    for _ in range(n_ops):
        key = rng.randint(1, key_pool)
        if rng.random() < 0.45:
            ops.append(LRUOperation("get", key))
        else:
            ops.append(LRUOperation("put", key, rng.randint(1, 100)))
    print(f"Generated {len(ops)} ops for capacity={capacity}, key_pool=1..{key_pool}")
    return ops


def print_lru_trace_table(operations: Sequence[LRUOperation]) -> pd.DataFrame:
    """Таблица операций для пошаговой отладки вашей реализации."""
    rows = [{"#": i + 1, "op": o.op, "key": o.key, "value": o.value} for i, o in enumerate(operations)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def plot_lru_access_pattern(operations: Sequence[LRUOperation], title: str = "LRU access pattern") -> None:
    """График обращений к ключам (какие ключи чаще трогают потоки операций)."""
    keys = [op.key for op in operations]
    counts = Counter(keys)
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    labels, values = zip(*items) if items else ([], [])

    plt.figure(figsize=(8, 4))
    plt.bar([str(k) for k in labels], values, color="teal", edgecolor="black")
    plt.xlabel("Key")
    plt.ylabel("Operations count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. REST API «Библиотека книг» — DTO, валидация, HTTP-шаблоны
# ---------------------------------------------------------------------------

@dataclass
class BookDTO:
    title: str
    author: str
    year: int
    genre: str
    id: int | None = None


def generate_sample_books(n: int = 5, seed: int = 42) -> list[BookDTO]:
    """Тестовые книги для POST/GET/PUT — не готовый Spring-сервис."""
    rng = random.Random(seed)
    genres = ["fiction", "science", "history", "fantasy", "biography"]
    authors = ["Tolstoy", "Dostoevsky", "Pushkin", "Bulgakov", "Gogol"]
    books = []
    for i in range(n):
        books.append(
            BookDTO(
                id=i + 1,
                title=f"Book {i + 1}",
                author=rng.choice(authors),
                year=rng.randint(1800, 2024),
                genre=rng.choice(genres),
            )
        )
    return books


def validate_book_dto(book: BookDTO | dict, current_year: int | None = None) -> tuple[bool, list[str]]:
    """
    Проверка полей DTO (год не в будущем, непустые строки).
    Используйте как эталон правил валидации в Java.
    """
    current_year = current_year or datetime.now().year
    errors: list[str] = []
    data = book if isinstance(book, dict) else asdict(book)

    for field in ("title", "author", "genre"):
        if not str(data.get(field, "")).strip():
            errors.append(f"{field} must not be empty")

    year = data.get("year")
    if not isinstance(year, int):
        errors.append("year must be integer")
    elif year > current_year:
        errors.append(f"year must not be in the future (>{current_year})")
    elif year < 0:
        errors.append("year must be non-negative")

    return len(errors) == 0, errors


def books_to_json(books: Sequence[BookDTO], indent: int = 2) -> str:
    """Сериализация списка книг в JSON для тела POST/PUT."""
    payload = [asdict(b) for b in books]
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def save_books_json(books: Sequence[BookDTO], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(books_to_json(books), encoding="utf-8")
    print(f"Saved {len(books)} books to {path}")
    return path


def print_curl_templates(base_url: str = "http://localhost:8080") -> None:
    """Шаблоны curl-запросов — копируйте и подставляйте свои данные."""
    templates = {
        "POST /books": f"""curl -X POST {base_url}/books \\
  -H "Content-Type: application/json" \\
  -d '{{"title":"War and Peace","author":"Tolstoy","year":1869,"genre":"fiction"}}'""",
        "GET /books": f'curl "{base_url}/books?author=Tolstoy&genre=fiction"',
        "GET /books/{{id}}": f"curl {base_url}/books/1",
        "PUT /books/{{id}}": f"""curl -X PUT {base_url}/books/1 \\
  -H "Content-Type: application/json" \\
  -d '{{"title":"Updated","author":"Tolstoy","year":1869,"genre":"classic"}}'""",
        "DELETE /books/{{id}}": f"curl -X DELETE {base_url}/books/1",
    }
    for name, cmd in templates.items():
        print(f"\n--- {name} ---\n{cmd}\n")


def print_rest_layer_skeleton() -> None:
    """Скелет слоёв Controller / Service / DTO — напоминание структуры, не код решения."""
    skeleton = """
    [Controller]  @RestController, @RequestMapping("/books")
                  POST/GET/PUT/DELETE + @Valid DTO + ResponseEntity

    [Service]     бизнес-логика, ConcurrentHashMap<Long, Book>

    [DTO]         BookRequest (ввод), BookResponse (вывод)

    [Exceptions]  @ControllerAdvice + @ExceptionHandler
                  BookNotFoundException, ValidationException
    """
    print(skeleton.strip())


# ---------------------------------------------------------------------------
# 4. Парсинг server.log — regex, генерация, визуализация (не готовый парсер)
# ---------------------------------------------------------------------------

LOG_LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"(?P<level>\w+) (?P<service>\w+) - (?P<message>.+)$"
)


def get_log_line_pattern() -> re.Pattern[str]:
    """Вернуть скомпилированный Pattern для формата server.log."""
    return LOG_LINE_PATTERN


def parse_log_line(line: str, pattern: re.Pattern[str] | None = None) -> dict[str, str] | None:
    """Разбор одной строки лога — для отладки regex, не замена Stream API в Java."""
    pattern = pattern or LOG_LINE_PATTERN
    m = pattern.match(line.strip())
    if not m:
        return None
    return m.groupdict()


def iter_log_lines(path: str | Path, encoding: str = "utf-8") -> Iterator[str]:
    """Построчное чтение файла (аналог Files.lines для быстрой проверки в Python)."""
    with Path(path).open(encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\n")


def preview_log_parsing(path: str | Path, n_lines: int = 10) -> pd.DataFrame:
    """Первые n строк: сработал ли regex."""
    rows = []
    for i, line in enumerate(iter_log_lines(path)):
        if i >= n_lines:
            break
        parsed = parse_log_line(line)
        rows.append({"line": line, "parsed": parsed is not None, **(parsed or {})})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def generate_sample_server_log(
    n_lines: int = 100,
    services: Sequence[str] | None = None,
    seed: int = 42,
    error_ratio: float = 0.25,
) -> list[str]:
    """Сгенерировать server.log для локальных тестов Java-парсера."""
    rng = random.Random(seed)
    services = list(services or ["UserService", "PaymentService", "AuthService", "DatabaseService"])
    levels_info = [
        ("INFO", "{service} - User {uid} logged in"),
        ("INFO", "{service} - Token expired for user {uid}"),
        ("ERROR", "{service} - Payment failed for user {uid}: timeout"),
        ("ERROR", "{service} - Connection pool exhausted"),
        ("ERROR", "{service} - Invalid credentials for user {uid}"),
    ]
    base_time = datetime(2024, 3, 15, 10, 0, 0)
    lines: list[str] = []
    for i in range(n_lines):
        ts = base_time + timedelta(seconds=i * rng.randint(5, 40))
        level, tmpl = rng.choice(levels_info)
        if rng.random() > error_ratio and level == "ERROR":
            level, tmpl = levels_info[0]
        svc = rng.choice(services)
        msg = tmpl.format(service=svc, uid=rng.randint(100, 999))
        lines.append(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {level} {svc} - {msg.split(' - ', 1)[-1]}")
    return lines


def save_server_log(lines: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(lines)} log lines to {path}")
    return path


def normalize_error_message(message: str) -> str:
    """Упростить текст ошибки (убрать id пользователя) — для группировки топ-сообщений."""
    msg = re.sub(r"user \d+", "user", message, flags=re.IGNORECASE)
    msg = re.sub(r"\b\d+\b", "", msg)
    return " ".join(msg.lower().split())


def demo_log_aggregation(log_path: str | Path) -> tuple[Counter[str], Counter[str]]:
    """
    Демо-агрегация ERROR (для сверки формата вывода).
    В экзамене реализуйте аналог на Java Stream API + Pattern/Matcher.
    """
    by_service: Counter[str] = Counter()
    by_message: Counter[str] = Counter()
    for line in iter_log_lines(log_path):
        parsed = parse_log_line(line)
        if not parsed or parsed["level"] != "ERROR":
            continue
        by_service[parsed["service"]] += 1
        by_message[normalize_error_message(parsed["message"])] += 1
    return by_service, by_message


def format_log_report(
    service_counts: Counter[str] | dict[str, int],
    message_counts: Counter[str] | dict[str, int],
    top_n: int = 3,
) -> str:
    """Формат эталонного вывода — сверьте с требованиями задания."""
    lines = ["Ошибки по сервисам:"]
    for svc, cnt in sorted(service_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{svc}: {cnt}")
    lines.append("")
    lines.append(f"Топ-{top_n} сообщений об ошибках:")
    for i, (msg, cnt) in enumerate(
        sorted(message_counts.items(), key=lambda x: (-x[1], x[0]))[:top_n], start=1
    ):
        lines.append(f"{i}. {msg} - {cnt} раз")
    return "\n".join(lines)


def plot_log_errors_by_service(service_counts: dict[str, int] | Counter[str], title: str = "Errors by service") -> None:
    """Bar chart ошибок по сервисам."""
    if not service_counts:
        print("No errors to plot")
        return
    items = sorted(service_counts.items(), key=lambda x: (-x[1], x[0]))
    labels, values = zip(*items)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="indianred", edgecolor="black")
    plt.xlabel("Service")
    plt.ylabel("ERROR count")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Теория: шпаргалки и быстрые проверки (SQL, Git, сети, ML-метрики)
# ---------------------------------------------------------------------------

def print_sql_cheatsheet() -> None:
    """Шаблоны SQL из типовых экзаменационных тем."""
    snippets = {
        "SELECT basics": "SELECT col1, col2 FROM table WHERE condition ORDER BY col1;",
        "JOIN": "SELECT a.*, b.name FROM a JOIN b ON a.id = b.a_id;",
        "GROUP BY": "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5;",
        "Window fn": "SELECT id, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM emp;",
        "Recursive CTE": "WITH RECURSIVE tree AS (SELECT id, parent_id FROM nodes WHERE parent_id IS NULL "
        "UNION ALL SELECT n.id, n.parent_id FROM nodes n JOIN tree t ON n.parent_id = t.id) SELECT * FROM tree;",
    }
    for title, sql in snippets.items():
        print(f"\n--- {title} ---\n{sql}")


def print_git_cheatsheet() -> None:
    """Git-команды из блока «Системы контроля версий»."""
    commands = {
        "История коммитов": "git log --oneline --graph",
        "Отмена изменений в файле": "git restore <file>  или  git checkout -- <file>",
        "Статус": "git status",
        "Diff": "git diff",
        "Staged diff": "git diff --staged",
    }
    for desc, cmd in commands.items():
        print(f"{desc}: {cmd}")


def print_network_reference() -> None:
    """Справочник по сетям (порты, протоколы, OSI)."""
    ref = {
        "HTTPS": "443",
        "HTTP": "80",
        "SSH": "22",
        "SFTP": "поверх SSH (22)",
        "DHCP": "автоназначение IP",
        "OSI levels": "7 (Physical … Application)",
        "HTTP idempotent": "GET, PUT, DELETE (HEAD, OPTIONS)",
    }
    for k, v in ref.items():
        print(f"{k}: {v}")


def print_docker_k8s_reference() -> None:
    print("Dockerfile: инструкции сборки образа (FROM, RUN, COPY, CMD)")
    print("docker build: сборка образа из Dockerfile")
    print("Orchestration: Kubernetes (k8s), Docker Swarm, Nomad")


def compute_classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Быстрый расчёт accuracy / precision / recall / F1 для теории ML."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == 1) if set(labels) <= {0, 1} else None

    if tp is not None:
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == p == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(y_true) if y_true else 0.0
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    else:
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
        metrics = {"accuracy": accuracy}

    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
    return metrics


def print_theory_topics_index() -> None:
    """Оглавление тем из файла ОМ_ГЭК.docx."""
    topics = [
        "1. Базы данных и SQL (PK, SELECT, ACID)",
        "2. Алгоритмы и Python (рекурсия, __init__, iterator, FIFO, декораторы)",
        "3. Linux (touch, rm, ls)",
        "4. Сети (HTTPS/443, DHCP, OSI, SFTP, идемпотентность HTTP)",
        "5. Архитектура (UML sequence, C4, CAP, микросервисы)",
        "6. Docker / оркестрация",
        "7. Git",
        "8. ML (RL, Recall, регрессия, adversarial attacks, feature engineering)",
        "9. Распределённые системы (Service Mesh, Kafka topic, масштабирование)",
        "10. ITSM/ITIL (SLA, incident)",
        "11. Cloud (PaaS, SaaS)",
        "12. Java практика: CompletableFuture, LRU, REST, log parsing",
    ]
    print("\n".join(topics))


# ---------------------------------------------------------------------------
# 6. Общие helpers
# ---------------------------------------------------------------------------

def setup_plot_style() -> None:
    """Единый стиль графиков для отчётов."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["font.size"] = 11


if __name__ == "__main__":
    setup_plot_style()
    print("=== OM/GEK Exam Utils Demo ===\n")

    nums = generate_random_int_list(10_000, seed=1)
    chunks = split_into_chunks(nums, 4)
    ref = reference_sum_of_squares(nums)
    print(f"Reference sum of squares (10k): {ref}")

    ops = generate_lru_test_sequence(capacity=3, n_ops=12)
    print_lru_trace_table(ops)

    books = generate_sample_books(3)
    ok, errs = validate_book_dto(books[0])
    print(f"Book validation: ok={ok}, errors={errs}")

    log_lines = generate_sample_server_log(20)
    log_path = Path(__file__).parent / "data" / "sample_server.log"
    save_server_log(log_lines, log_path)
    by_svc, by_msg = demo_log_aggregation(log_path)
    print("\n" + format_log_report(by_svc, by_msg))

    print_theory_topics_index()
