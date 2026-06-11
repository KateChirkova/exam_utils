"""
Примеры использования утилит для подготовки к ГЭК (ОМ, IT-сервисы).
Запуск: python exam_examples.py
"""

from pathlib import Path

from utils import *
from llm_helpers import build_concept_prompt, build_code_review_prompt, load_llm_config

DATA_DIR = Path(__file__).parent / "data"


def example_1_parallel_processing():
    """Задача 1: тестовые данные и эталон для CompletableFuture."""
    print("EXAMPLE 1: Parallel processing helpers")

    n = 100_000  # на экзамене в Java — 1_000_000
    data = generate_random_int_list(n, seed=42)
    chunks = split_into_chunks(data, 4)

    partial_sums = [reference_sum_of_squares(c) for c in chunks]
    total = sum(partial_sums)
    expected = reference_sum_of_squares(data)
    assert total == expected, "Chunk sums mismatch!"

    benchmark_call("sum_of_squares (Python ref)", lambda: reference_sum_of_squares(data))

    chunk_sizes = [len(c) for c in chunks]
    print(f"Total={total}, chunks={chunk_sizes}")
    plot_parallel_chunks(chunk_sizes, partial_sums)


def example_2_lru_cache_trace():
    """Задача 2: трассировка операций LRU (реализация — в Java)."""
    print("EXAMPLE 2: LRU test sequence")

    ops = generate_lru_test_sequence(capacity=3, n_ops=15, key_pool=5)
    print_lru_trace_table(ops)
    plot_lru_access_pattern(ops)


def example_3_rest_api_helpers():
    """Задача 3: DTO, валидация, curl-шаблоны."""
    print("EXAMPLE 3: REST API helpers")

    books = generate_sample_books(5)
    for b in books:
        ok, errs = validate_book_dto(b)
        print(f"  {b.title}: valid={ok}, errors={errs}")

    save_books_json(books, DATA_DIR / "sample_books.json")
    print_rest_layer_skeleton()
    print_curl_templates()


def example_4_log_parsing():
    """Задача 4: генерация log, regex, формат отчёта."""
    print("EXAMPLE 4: Log parsing helpers")

    log_path = DATA_DIR / "sample_server.log"
    if not log_path.exists():
        lines = generate_sample_server_log(80, seed=7)
        save_server_log(lines, log_path)

    preview_log_parsing(log_path, n_lines=5)

    by_service, by_message = demo_log_aggregation(log_path)
    report = format_log_report(by_service, by_message, top_n=3)
    print("\n" + report)

    plot_log_errors_by_service(by_service)


def example_5_theory_cheatsheets():
    """Теоретический блок из ОМ_ГЭК.docx."""
    print("EXAMPLE 5: Theory cheatsheets")

    print_theory_topics_index()
    print("\n--- SQL ---")
    print_sql_cheatsheet()
    print("\n--- Git ---")
    print_git_cheatsheet()
    print("\n--- Networks ---")
    print_network_reference()
    print("\n--- Docker/K8s ---")
    print_docker_k8s_reference()

    print("\n--- ML metrics demo ---")
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
    compute_classification_metrics(y_true, y_pred)


def example_6_llm_prompts():
    """Шаблоны промптов для LLM (без вызова API)."""
    print("EXAMPLE 6: LLM prompt templates")

    print(build_concept_prompt("CompletableFuture в Java"))
    print("\n---")
    print(build_code_review_prompt(
        "public class LRUCache<K,V> { /* ваш черновик */ }",
        language="java",
    ))
    cfg = load_llm_config()
    print(f"\nLLM config: provider={cfg.provider}, model={cfg.model}")
    print("Для вызова API: from llm_helpers import explain_concept; explain_concept('CAP theorem')")


EXAMPLES = [
    ("Parallel processing", example_1_parallel_processing),
    ("LRU trace", example_2_lru_cache_trace),
    ("REST helpers", example_3_rest_api_helpers),
    ("Log parsing", example_4_log_parsing),
    ("Theory cheatsheets", example_5_theory_cheatsheets),
    ("LLM prompts", example_6_llm_prompts),
]


if __name__ == "__main__":
    setup_plot_style()
    print("OM/GEK EXAM PREPARATION EXAMPLES\n")

    for name, func in EXAMPLES:
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
        try:
            func()
        except Exception as exc:
            print(f"Skipped/error in {name}: {exc}")

    print("\nDone. Use individual example functions during preparation.")
