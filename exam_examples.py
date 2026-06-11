"""
Примеры использования утилит.
Ориентир: https://github.com/Moldarus/exam-preparation-utils
"""

from utils import *
import pandas as pd
import numpy as np


def example_1_eda_and_visualization():
    print("EXAMPLE 1: EDA AND VISUALIZATION")
    setup_plot_style()
    df = generate_sample_data('mixed', n_samples=500)
    quick_eda(df, target_col='target')
    quick_visualization_report(df, target_col='target')
    return df


def example_2_data_preprocessing():
    print("EXAMPLE 2: DATA PREPROCESSING")
    df = pd.DataFrame({
        'age': [25, 30, np.nan, 35, 40, np.nan, 45],
        'salary': [50000, 60000, 55000, np.nan, 70000, 65000, 80000],
        'city': ['Moscow', 'SPb', 'Moscow', 'Kazan', np.nan, 'Moscow', 'SPb'],
        'target': [0, 1, 0, 1, 0, 1, 0]
    })
    preview_df(df)
    df_clean = handle_missing_values(df, strategy='auto')
    df_encoded, _ = encode_categorical(df_clean, method='auto')
    df_scaled, _ = scale_features(df_encoded, target_col='target')
    return df_scaled


def example_3_clustering():
    print("EXAMPLE 3: CLUSTERING")
    df = generate_sample_data('clustering', n_samples=300)
    X = df[['x', 'y', 'z']].values
    optimal_k = kmeans_elbow_method(X, max_k=8)
    kmeans_labels, _ = perform_clustering(X, n_clusters=optimal_k)
    plot_clusters_2d(X, kmeans_labels, "K-Means Clustering", x_col=0, y_col=1)
    df['kmeans_cluster'] = kmeans_labels
    plot_parallel_coordinates(df, 'kmeans_cluster', ['x', 'y', 'z'])
    return df


def example_4_classification():
    print("EXAMPLE 4: CLASSIFICATION")
    df = generate_sample_data('classification', n_samples=500)
    df_clean = handle_missing_values(df)
    df_encoded, _ = encode_categorical(df_clean)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = X_test.copy()
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    results = train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
    best_model = max(results, key=lambda x: results[x]['f1'])
    y_pred = results[best_model]['model'].predict(X_test_scaled)
    plot_confusion_matrix_custom(y_test, y_pred, f"Confusion Matrix - {best_model}")
    return results


def example_5_regression():
    print("EXAMPLE 5: REGRESSION")
    df = generate_sample_data('regression', n_samples=500)
    df_clean = handle_missing_values(df)
    df_encoded, _ = encode_categorical(df_clean)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = X_test.copy()
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    results = train_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
    best_model = max(results, key=lambda x: results[x]['r2'])
    y_pred = results[best_model]['model'].predict(X_test_scaled)
    plot_predictions(y_test, y_pred, f"Predictions - {best_model}")
    return results


def example_6_anomaly_detection():
    print("EXAMPLE 6: ANOMALY DETECTION")
    np.random.seed(42)
    data = np.concatenate([np.random.normal(100, 10, 100), np.random.uniform(50, 150, 5)])
    df = pd.DataFrame({'value': data})
    _, _, bounds = detect_anomalies_iqr(df, 'value')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['value'], bins=20, alpha=0.7)
    plt.axvline(bounds[0], color='r', linestyle='--')
    plt.axvline(bounds[1], color='r', linestyle='--')
    plt.subplot(1, 2, 2)
    plt.boxplot(df['value'])
    plt.tight_layout()
    plt.show()
    return df


def example_7_full_pipeline():
    print("EXAMPLE 7: FULL ML PIPELINE")
    df = generate_sample_data('classification', n_samples=500)
    return create_ml_pipeline(df, 'target')


def example_8_llm_connection():
    print("EXAMPLE 8: LLM connection")
    from llm_helpers import load_llm_config, chat
    cfg = load_llm_config()
    print(f"provider={cfg.provider}, model={cfg.model}")
    print('chat([{"role": "user", "content": "ваш текст"}])')


if __name__ == "__main__":
    setup_plot_style()
    examples = [
        ("EDA and Visualization", example_1_eda_and_visualization),
        ("Data Preprocessing", example_2_data_preprocessing),
        ("Clustering", example_3_clustering),
        ("Classification", example_4_classification),
        ("Regression", example_5_regression),
        ("Anomaly Detection", example_6_anomaly_detection),
        ("Full Pipeline", example_7_full_pipeline),
        ("LLM connection", example_8_llm_connection),
    ]
    for name, func in examples:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
