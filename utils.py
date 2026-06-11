"""
Утилиты для практических заданий на экзамене.
Ориентир: https://github.com/Moldarus/exam-preparation-utils

Назначение — заменить рутину: загрузка данных, EDA, графики, предобработка,
обучение моделей, работа с файлами. Не решает конкретную задачу целиком.
"""

from __future__ import annotations

import re
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Рутинные операции (файлы, таймер, разбиение, группировки)
# =============================================================================

def setup_plot_style(figsize: tuple[int, int] = (10, 5)) -> None:
    """Единый стиль графиков — один вызов в начале работы."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = 11


def save_figure(path: str | Path, dpi: int = 120) -> Path:
    """Сохранить текущую фигуру (вместо ручного plt.savefig)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved: {path}")
    return path


def preview_df(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Быстрый просмотр: shape, dtypes, head."""
    print(f"Shape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}\n")
    print(df.head(n))
    return df.head(n)


def load_table(path: str | Path) -> pd.DataFrame:
    """Загрузить CSV / Excel / JSON без выбора метода вручную."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported format: {suffix}")


def read_text_lines(path: str | Path, encoding: str = "utf-8") -> list[str]:
    """Построчное чтение текстового файла."""
    return Path(path).read_text(encoding=encoding).splitlines()


@contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """Контекстный менеджер замера времени в мс."""
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[{label}] {elapsed:.2f} ms")


def benchmark_call(label: str, func: Callable[[], Any], repeat: int = 1) -> dict[str, Any]:
    """Замер callable; удобно сравнивать варианты реализации."""
    times_ms: list[float] = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times_ms.append((time.perf_counter() - t0) * 1000)
    stats = {
        "label": label,
        "result": result,
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }
    print(f"[{label}] mean={stats['mean_ms']:.2f} ms")
    return stats


def split_into_chunks(data: Sequence, n_chunks: int) -> list[list]:
    """Разбить последовательность на n частей."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    size = len(data)
    base, extra = divmod(size, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        end = start + base + (1 if i < extra else 0)
        chunks.append(list(data[start:end]))
        start = end
    return chunks


def group_and_count(df: pd.DataFrame, column: str, sort: bool = True) -> pd.Series:
    """value_counts + сортировка — частая операция в аналитических задачах."""
    counts = df[column].value_counts()
    return counts.sort_values(ascending=False) if sort else counts


def top_n_counts(
    counts: Counter | pd.Series | dict,
    n: int = 10,
    title: str = "Top counts",
) -> pd.DataFrame:
    """Топ-N значений в таблице (для отчёта или сверки)."""
    if isinstance(counts, Counter):
        items = counts.most_common(n)
    elif isinstance(counts, pd.Series):
        items = list(counts.head(n).items())
    else:
        items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))[:n]
    df = pd.DataFrame(items, columns=["value", "count"])
    print(f"--- {title} ---")
    print(df.to_string(index=False))
    return df


def extract_with_regex(
    lines: Sequence[str],
    pattern: str | re.Pattern,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Применить regex к списку строк → DataFrame (отладка парсинга)."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    rows = []
    for line in lines:
        m = compiled.search(line.strip())
        if m:
            rows.append(m.groupdict() if m.groupdict() else {"match": m.group(0)})
    df = pd.DataFrame(rows)
    if fields:
        df = df[[c for c in fields if c in df.columns]]
    print(f"Matched {len(df)} / {len(lines)} lines")
    return df


# =============================================================================
# 1. EDA
# =============================================================================

def quick_eda(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """Быстрый EDA: shape, пропуски, describe, целевая переменная."""
    print("=" * 50, "BASIC INFO", "=" * 50, sep="\n")
    print(f"Shape: {df.shape}\nColumns: {list(df.columns)}\n\n{df.dtypes}")

    print("\n" + "=" * 50, "MISSING VALUES", "=" * 50, sep="\n")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({"Missing": missing, "Pct": (missing / len(df) * 100).round(2)})
    print(missing_df[missing_df["Missing"] > 0])

    print("\n" + "=" * 50, "STATISTICS", "=" * 50, sep="\n")
    print(df.describe(include="all"))

    if target_col and target_col in df.columns:
        print("\n" + "=" * 50, f"TARGET: {target_col}", "=" * 50, sep="\n")
        col = df[target_col]
        if pd.api.types.is_numeric_dtype(col):
            print(f"mean={col.mean():.2f}, median={col.median():.2f}, std={col.std():.2f}")
        else:
            print(col.value_counts().head(10))

    return missing_df


# =============================================================================
# 2. Визуализация (упрощённые обёртки)
# =============================================================================

def plot_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    figsize: tuple[int, int] = (15, 10),
    bins: int = 30,
) -> None:
    """Гистограммы числовых колонок в одной фигуре."""
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns")
        return

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=bins, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)

    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple[int, int] = (10, 8)) -> pd.DataFrame | None:
    """Тепловая карта корреляций."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns")
        return None

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    return corr


def plot_bar_counts(
    data: Counter | pd.Series | dict,
    title: str = "Counts",
    top_n: int | None = None,
    figsize: tuple[int, int] = (10, 5),
    horizontal: bool = False,
) -> None:
    """Bar chart из value_counts / Counter / dict — без ручной настройки осей."""
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
    fn = plt.barh if horizontal else plt.bar
    fn(labels, values, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_box_by_category(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    title: str | None = None,
) -> None:
    """Boxplot числового признака по категориям."""
    plt.figure(figsize=(10, 5))
    df.boxplot(column=numeric_col, by=category_col)
    plt.title(title or f"{numeric_col} by {category_col}")
    plt.suptitle("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_scatter_colored(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None = None,
    title: str = "Scatter",
) -> None:
    """Scatter с опциональной раскраской по категории."""
    plt.figure(figsize=(8, 6))
    if color_col and color_col in df.columns:
        for label, group in df.groupby(color_col):
            plt.scatter(group[x], group[y], label=str(label), alpha=0.6)
        plt.legend()
    else:
        plt.scatter(df[x], df[y], alpha=0.6)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def quick_visualization_report(df: pd.DataFrame, target_col: str | None = None) -> None:
    """Один вызов → корреляции + распределения + анализ целевой."""
    print("1. Correlation matrix")
    plot_correlation_matrix(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("2. Distributions")
        plot_distributions(df, numeric_cols[: min(6, len(numeric_cols))])

    if target_col and target_col in df.columns:
        print(f"3. Target: {target_col}")
        col = df[target_col]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if pd.api.types.is_numeric_dtype(col):
            axes[0].hist(col.dropna(), bins=30, edgecolor="black")
            axes[1].boxplot(col.dropna())
        else:
            counts = col.value_counts().head(10)
            axes[0].bar(counts.index.astype(str), counts.values)
            plot_bar_counts(counts, title=f"{target_col} counts", top_n=10)
            plt.close(fig)
            return
        axes[0].set_title(f"Distribution of {target_col}")
        axes[1].set_title(f"Boxplot of {target_col}")
        plt.tight_layout()
        plt.show()


def plot_parallel_coordinates(df: pd.DataFrame, class_col: str, cols: list[str] | None = None) -> None:
    """Параллельные координаты."""
    from pandas.plotting import parallel_coordinates

    cols = cols or [c for c in df.select_dtypes(include=[np.number]).columns if c != class_col]
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df[cols + [class_col]], class_col, colormap="viridis")
    plt.title("Parallel Coordinates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Предобработка
# =============================================================================

def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """Заполнение пропусков: auto (медиана/мода), drop, zero."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0:
            continue
        if strategy == "auto":
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode = df_clean[col].mode()
                df_clean[col] = df_clean[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
        elif strategy == "drop":
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == "zero":
            df_clean[col] = df_clean[col].fillna(0)

    print(f"Missing: {df.isnull().sum().sum()} → {df_clean.isnull().sum().sum()}")
    return df_clean


def encode_categorical(df: pd.DataFrame, method: str = "auto") -> tuple[pd.DataFrame, dict]:
    """One-Hot / Label encoding категориальных колонок."""
    df_encoded = df.copy()
    encoders: dict = {}
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        if method == "auto":
            use_onehot = df_encoded[col].nunique() <= 10
        else:
            use_onehot = method == "onehot"

        if use_onehot:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            encoders[col] = "onehot"
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    print(f"Encoded {len(cat_cols)} categorical columns")
    return df_encoded, encoders


def scale_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler / MinMaxScaler для числовых признаков."""
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [c for c in df.columns if c != target_col] if target_col else df.columns.tolist()
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Scaled {len(numeric_cols)} columns ({method})")
    return df_scaled, scaler


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any | None]:
    """Типовой пайплайн: пропуски → encoding → split → scale."""
    df_clean = handle_missing_values(df)
    df_enc, _ = encode_categorical(df_clean)
    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = None
    if scale:
        X_train, scaler = scale_features(X_train)
        X_test = X_test.copy()
        num = X_test.select_dtypes(include=[np.number]).columns
        X_test[num] = scaler.transform(X_test[num])

    return X_train, X_test, y_train, y_test, scaler


# =============================================================================
# 4. Кластеризация
# =============================================================================

def kmeans_elbow_method(X: np.ndarray, max_k: int = 10, random_state: int = 42) -> int:
    """Метод локтя + график inertia."""
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, "bo-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    if len(inertias) > 2:
        diffs2 = np.diff(np.diff(inertias))
        optimal_k = int(np.argmin(diffs2) + 2)
        print(f"Suggested k: {optimal_k}")
        return optimal_k
    return 3


def perform_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """K-Means + DBSCAN за один вызов."""
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"K-Means: {n_clusters} clusters | DBSCAN: {n_db} clusters")
    return kmeans_labels, dbscan_labels


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Clusters",
    x_col: int = 0,
    y_col: int = 1,
) -> None:
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(X[:, x_col], X[:, y_col], c=labels, cmap="viridis", alpha=0.6)
    plt.colorbar(sc)
    plt.xlabel(f"Feature {x_col}")
    plt.ylabel(f"Feature {y_col}")
    plt.title(title)
    plt.show()


# =============================================================================
# 5. Классификация
# =============================================================================

def train_classification_models(X_train, X_test, y_train, y_test) -> dict:
    """Обучить и сравнить 3 классификатора с метриками."""
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        res = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "auc": roc_auc_score(y_test, proba) if proba is not None and len(np.unique(y_test)) == 2 else None,
        }
        results[name] = res
        print(f"{name}: acc={res['accuracy']:.4f}, f1={res['f1']:.4f}")
    return results


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()


def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    score = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={score:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# 6. Регрессия
# =============================================================================

def train_regression_models(X_train, X_test, y_train, y_test) -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": float(np.sqrt(mse)),
            "r2": r2_score(y_test, y_pred),
        }
        print(f"{name}: MAE={results[name]['mae']:.4f}, R²={results[name]['r2']:.4f}")
    return results


def plot_predictions(y_true, y_pred, title: str = "Predictions vs Actual") -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lo, hi = min(y_true), max(y_true)
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.show()


# =============================================================================
# 7. Аномалии
# =============================================================================

def detect_anomalies_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - multiplier * IQR, Q3 + multiplier * IQR
    anomalies = df[(df[column] < lo) | (df[column] > hi)]
    print(f"{column}: bounds [{lo:.2f}, {hi:.2f}], anomalies={len(anomalies)}")
    return anomalies, df[(df[column] >= lo) & (df[column] <= hi)], (lo, hi)


def detect_anomalies_zscore(df: pd.DataFrame, column: str, threshold: float = 3):
    z = np.abs((df[column] - df[column].mean()) / df[column].std())
    anomalies = df[z > threshold]
    print(f"{column}: z>{threshold}, anomalies={len(anomalies)}")
    return anomalies, df[z <= threshold]


# =============================================================================
# 8. Feature engineering / selection
# =============================================================================

def create_interaction_features(df: pd.DataFrame, feature_pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    df_new = df.copy()
    if feature_pairs is None:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num].corr().abs()
        feature_pairs = [
            (num[i], num[j])
            for i in range(len(num))
            for j in range(i + 1, len(num))
            if corr.iloc[i, j] > 0.5
        ][:10]
    for c1, c2 in feature_pairs:
        if c1 in df.columns and c2 in df.columns:
            df_new[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    return df_new


def select_features_by_correlation(df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> pd.DataFrame:
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    corr = df[num].corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    print(f"Dropping correlated: {drop}")
    return df.drop(columns=drop)


def cross_validate_model(model, X, y, cv: int = 5, scoring: str = "auto") -> np.ndarray:
    if scoring == "auto":
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "f1_weighted"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"CV {scoring}: {scores.mean():.4f} ± {scores.std() * 2:.4f}")
    return scores


# =============================================================================
# 9. Генерация тестовых данных
# =============================================================================

def generate_sample_data(dataset_type: str = "regression", n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Синтетический датасет для отладки пайплайна."""
    np.random.seed(random_state)

    if dataset_type == "regression":
        X = np.random.randn(n_samples, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, 6)])
        df["target"] = y
    elif dataset_type == "classification":
        X = np.random.randn(n_samples, 5)
        prob = 1 / (1 + np.exp(-(2 * X[:, 0] + X[:, 1])))
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, 6)])
        df["target"] = (prob > 0.5).astype(int)
    elif dataset_type == "clustering":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=3, random_state=random_state)
        df = pd.DataFrame(X, columns=["x", "y", "z"])
        df["cluster"] = y
    else:
        df = pd.DataFrame({
            "n1": np.random.randn(n_samples),
            "n2": np.random.exponential(2, n_samples),
            "cat": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randn(n_samples),
        })
        df.loc[df.sample(frac=0.05, random_state=random_state).index, "n1"] = np.nan

    print(f"Generated '{dataset_type}' dataset: {df.shape}")
    return df


# =============================================================================
# 10. ML pipeline (склейка рутины)
# =============================================================================

def create_ml_pipeline(df: pd.DataFrame, target_col: str, problem_type: str = "auto"):
    """Полный цикл: preprocess → train → метрики → графики."""
    print("ML Pipeline")
    if problem_type == "auto":
        problem_type = "regression" if df[target_col].nunique() > 10 and pd.api.types.is_numeric_dtype(df[target_col]) else "classification"
    print(f"Type: {problem_type}")

    X_train, X_test, y_train, y_test, _ = prepare_xy(df, target_col)

    if problem_type == "classification":
        results = train_classification_models(X_train, X_test, y_train, y_test)
        best = max(results, key=lambda k: results[k]["f1"])
        y_pred = results[best]["model"].predict(X_test)
        plot_confusion_matrix(y_test, y_pred, f"CM — {best}")
        if results[best]["auc"] is not None:
            plot_roc_curve(y_test, results[best]["model"].predict_proba(X_test)[:, 1], f"ROC — {best}")
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test)
        best = max(results, key=lambda k: results[k]["r2"])
        plot_predictions(y_test, results[best]["model"].predict(X_test), f"Pred — {best}")

    return results[best]["model"], results, X_test, y_test


if __name__ == "__main__":
    setup_plot_style()
    df = generate_sample_data("mixed", 300)
    quick_eda(df, "target")
    top_n_counts(group_and_count(df, "cat"), n=5, title="Categories")
