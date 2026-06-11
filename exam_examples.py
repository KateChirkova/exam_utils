"""
Примеры визуализаций и подключения LLM.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import *


def example_1_visualization_report():
    print("EXAMPLE 1: Visualization report")
    setup_plot_style()
    np.random.seed(42)
    df = pd.DataFrame({
        'f1': np.random.randn(200),
        'f2': np.random.randn(200) * 2,
        'f3': np.random.randn(200) + 1,
        'target': np.random.choice([0, 1], 200),
    })
    preview_df(df)
    quick_visualization_report(df, target_col='target')
    return df


def example_2_bar_and_counts():
    print("EXAMPLE 2: Bar counts")
    plot_bar_counts({'positive': 45, 'negative': 30, 'neutral': 25}, title='Sentiment')


def example_3_confusion_matrix():
    print("EXAMPLE 3: Confusion matrix")
    X, y = make_classification(n_samples=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_confusion_matrix_custom(y_test, y_pred, title="Confusion Matrix")
    if hasattr(model, 'predict_proba'):
        plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])


def example_4_regression_plot():
    print("EXAMPLE 4: Predictions plot")
    y_true = np.linspace(0, 10, 50)
    y_pred = y_true + np.random.randn(50) * 0.5
    plot_predictions(y_true, y_pred)


def example_5_llm_client():
    print("EXAMPLE 5: LLM client")
    from llm_helpers import load_llm_config, get_llm_client
    cfg = load_llm_config()
    print(f"provider={cfg.provider}, model={cfg.model}")
    client = get_llm_client(cfg)
    print("client ready — вызов API в вашем коде")


if __name__ == "__main__":
    examples = [
        ("Visualization report", example_1_visualization_report),
        ("Bar counts", example_2_bar_and_counts),
        ("Confusion matrix", example_3_confusion_matrix),
        ("Regression plot", example_4_regression_plot),
        ("LLM client", example_5_llm_client),
    ]
    for name, func in examples:
        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
