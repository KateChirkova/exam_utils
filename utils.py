from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def setup_plot_style(figsize=(10, 5)):
    """Единый стиль графиков — один вызов в начале работы."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 11


def save_figure(path, dpi=120):
    """Сохранить текущую фигуру."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved: {path}")
    return path


def load_table(path):
    """CSV / Excel / JSON — без выбора read_* вручную."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if path.suffix.lower() == '.json':
        return pd.read_json(path)
    raise ValueError(f"Unsupported: {path.suffix}")


def preview_df(df, n=5):
    """Shape, dtypes, head."""
    print(f"Shape: {df.shape}\n{df.dtypes}\n")
    print(df.head(n))
    return df.head(n)


def plot_bar_counts(data, title='Counts', top_n=None, figsize=(10, 5), horizontal=False):
    """Bar chart из Series, dict или Counter."""
    from collections import Counter
    if isinstance(data, Counter):
        items = data.most_common(top_n)
        labels, values = zip(*items) if items else ([], [])
    elif isinstance(data, pd.Series):
        s = data.head(top_n) if top_n else data
        labels, values = s.index.tolist(), s.values.tolist()
    else:
        items = sorted(data.items(), key=lambda x: -x[1])
        if top_n:
            items = items[:top_n]
        labels, values = zip(*items) if items else ([], [])
    plt.figure(figsize=figsize)
    (plt.barh if horizontal else plt.bar)(labels, values, edgecolor='black', alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_distributions(df, numeric_cols=None, figsize=(15, 10), bins=30):
    """Гистограммы числовых колонок."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns")
        return

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        axes[i].set_title(col)

    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """Тепловая карта корреляций."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns")
        return None

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    return corr


def plot_clusters_2d(X, labels, title="Clustering Results", x_col=0, y_col=1):
    """Scatter кластеров в 2D."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, x_col], X[:, y_col], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel(f'Feature {x_col}')
    plt.ylabel(f'Feature {y_col}')
    plt.title(title)
    plt.show()


def plot_confusion_matrix_custom(y_true, y_pred, title="Confusion Matrix"):
    """Матрица ошибок (confusion matrix) — heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    return cm


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """ROC-кривая с AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    """Scatter: факт vs прогноз."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lo, hi = min(y_true), max(y_true)
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_parallel_coordinates(df, class_col, cols=None):
    """Параллельные координаты."""
    from pandas.plotting import parallel_coordinates

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if class_col in cols:
            cols.remove(class_col)

    plt.figure(figsize=(12, 6))
    parallel_coordinates(df[cols + [class_col]], class_col, colormap='viridis')
    plt.title('Parallel Coordinates')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def quick_visualization_report(df, target_col=None):
    """Один вызов: корреляции + распределения + целевая переменная."""
    print("1. Correlation Matrix")
    plot_correlation_matrix(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("2. Distributions")
        plot_distributions(df, numeric_cols[:min(6, len(numeric_cols))])

    if target_col and target_col in df.columns:
        print(f"3. Target: {target_col}")
        col = df[target_col]
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        if pd.api.types.is_numeric_dtype(col):
            col.hist(bins=30, edgecolor='black')
            plt.title(f'Distribution of {target_col}')
        else:
            counts = col.value_counts().head(10)
            plt.bar(counts.index.astype(str), counts.values)
            plt.xticks(rotation=45)
            plt.title(f'Counts of {target_col}')
        plt.subplot(1, 2, 2)
        if pd.api.types.is_numeric_dtype(col):
            plt.boxplot(col.dropna())
            plt.title(f'Boxplot of {target_col}')
        plt.tight_layout()
        plt.show()
