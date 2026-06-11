"""
Примеры использования утилит.
Ориентир: https://github.com/Moldarus/exam-preparation-utils
"""

from utils import *


def example_1_eda_and_visualization():
    print("EXAMPLE 1: EDA + visualization")
    df = generate_sample_data("mixed", n_samples=500)
    quick_eda(df, target_col="target")
    quick_visualization_report(df, target_col="target")
    return df


def example_2_data_preprocessing():
    print("EXAMPLE 2: Preprocessing")
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 35, 40],
        "salary": [50_000, 60_000, 55_000, np.nan, 70_000],
        "city": ["Moscow", "SPb", "Moscow", "Kazan", "Moscow"],
        "target": [0, 1, 0, 1, 0],
    })
    preview_df(df)
    df_clean = handle_missing_values(df)
    df_enc, _ = encode_categorical(df_clean)
    df_scaled, _ = scale_features(df_enc, target_col="target")
    return df_scaled


def example_3_clustering():
    print("EXAMPLE 3: Clustering")
    df = generate_sample_data("clustering", n_samples=300)
    X = df[["x", "y", "z"]].values
    k = kmeans_elbow_method(X, max_k=8)
    km_labels, db_labels = perform_clustering(X, n_clusters=k)
    plot_clusters_2d(X, km_labels)
    df["kmeans"] = km_labels
    plot_parallel_coordinates(df, "kmeans", ["x", "y", "z"])
    return df


def example_4_classification():
    print("EXAMPLE 4: Classification")
    df = generate_sample_data("classification", n_samples=500)
    X_train, X_test, y_train, y_test, _ = prepare_xy(df, "target")
    results = train_classification_models(X_train, X_test, y_train, y_test)
    best = max(results, key=lambda k: results[k]["f1"])
    y_pred = results[best]["model"].predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best)
    return results


def example_5_regression():
    print("EXAMPLE 5: Regression")
    df = generate_sample_data("regression", n_samples=500)
    X_train, X_test, y_train, y_test, _ = prepare_xy(df, "target")
    results = train_regression_models(X_train, X_test, y_train, y_test)
    best = max(results, key=lambda k: results[k]["r2"])
    plot_predictions(y_test, results[best]["model"].predict(X_test), best)
    return results


def example_6_anomaly_detection():
    print("EXAMPLE 6: Anomalies")
    data = np.concatenate([np.random.normal(100, 10, 100), np.random.uniform(30, 170, 8)])
    df = pd.DataFrame({"value": data})
    anomalies, normal, bounds = detect_anomalies_iqr(df, "value")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df["value"], bins=20, alpha=0.7)
    plt.axvline(bounds[0], color="r", linestyle="--")
    plt.axvline(bounds[1], color="r", linestyle="--")
    plt.subplot(1, 2, 2)
    plt.boxplot(df["value"])
    plt.tight_layout()
    plt.show()
    return df


def example_7_full_pipeline():
    print("EXAMPLE 7: Full ML pipeline")
    df = generate_sample_data("classification", n_samples=400)
    return create_ml_pipeline(df, "target")


def example_8_routine_helpers():
    print("EXAMPLE 8: Routine helpers")
    data = list(range(20))
    chunks = split_into_chunks(data, 4)
    print("Chunks:", [len(c) for c in chunks])

    with timer("sum"):
        _ = sum(x * x for x in data)

    lines = ["[2024-01-01 10:00:00] ERROR Svc - timeout", "[2024-01-01 10:01:00] INFO Svc - ok"]
    pattern = r"\[(?P<ts>[^\]]+)\] (?P<level>\w+) (?P<svc>\w+) - (?P<msg>.+)"
    parsed = extract_with_regex(lines, pattern)
    preview_df(parsed)

    plot_bar_counts({"a": 5, "b": 12, "c": 3}, title="Demo counts", top_n=3)
    return parsed


def example_9_llm_connection():
    print("EXAMPLE 9: LLM connection (no prompts)")
    from llm_helpers import load_llm_config, check_connection

    cfg = load_llm_config()
    print(f"provider={cfg.provider}, model={cfg.model}, key_set={bool(cfg.api_key)}")
    print("To call LLM, pass your own messages:")
    print('  from llm_helpers import chat')
    print('  chat([{"role": "user", "content": "ваш вопрос"}])')
    if cfg.api_key or cfg.provider == "ollama":
        check_connection(cfg.provider)


EXAMPLES = [
    ("EDA & visualization", example_1_eda_and_visualization),
    ("Preprocessing", example_2_data_preprocessing),
    ("Clustering", example_3_clustering),
    ("Classification", example_4_classification),
    ("Regression", example_5_regression),
    ("Anomaly detection", example_6_anomaly_detection),
    ("Full pipeline", example_7_full_pipeline),
    ("Routine helpers", example_8_routine_helpers),
    ("LLM connection", example_9_llm_connection),
]


if __name__ == "__main__":
    setup_plot_style()
    print("EXAM UTILS — EXAMPLES\n")
    for name, func in EXAMPLES:
        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        try:
            func()
        except Exception as exc:
            print(f"Error: {exc}")
