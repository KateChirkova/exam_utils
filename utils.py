"""
Утилиты для практических заданий на экзамене.
Ориентир: https://github.com/Moldarus/exam-preparation-utils

Сокращают рутину: EDA, визуализация, предобработка, ML-графики.
Не заменяют библиотеки и не решают задачу целиком.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings

warnings.filterwarnings('ignore')


# --- Визуализация и загрузка: короткие обёртки ---

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


#EDA (Разведочный анализ данных)

def quick_eda(df, target_col=None):
    """
    Быстрый EDA датафрейма
    Пример:
        df = pd.read_csv('data.csv')
        report = quick_eda(df, target_col='price')
    """
    print("=" * 50)
    print("BASIC INFO")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Missing'] > 0])

    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(df.describe())

    if target_col and target_col in df.columns:
        print("\n" + "=" * 50)
        print(f"TARGET ANALYSIS: {target_col}")
        print("=" * 50)
        if df[target_col].dtype in ['int64', 'float64']:
            print(f"Mean: {df[target_col].mean():.2f}")
            print(f"Median: {df[target_col].median():.2f}")
            print(f"Std: {df[target_col].std():.2f}")
        else:
            print(f"Value counts:\n{df[target_col].value_counts()}")

    return missing_df


def plot_distributions(df, numeric_cols=None, figsize=(15, 10)):
    """
    Построение графиков распределений
    Пример:
        plot_distributions(df, numeric_cols=['age', 'salary', 'price'])
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    # Удаляем лишние подграфики
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Матрица корреляций
    Пример:
        plot_correlation_matrix(df)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        return corr
    else:
        print("Not enough numeric columns for correlation")
        return None


#Предобработка данных

def handle_missing_values(df, strategy='auto'):
    """
    Обработка пропусков
    Пример:
        df_clean = handle_missing_values(df, strategy='auto')
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if strategy == 'auto':
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Для числовых - медиана
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Для категориальных - мода
                    df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown',
                                         inplace=True)
            elif strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
            elif strategy == 'zero':
                df_clean[col].fillna(0, inplace=True)

    print(f"Missing values handled. Original: {df.isnull().sum().sum()}, Now: {df_clean.isnull().sum().sum()}")
    return df_clean


def encode_categorical(df, method='auto'):
    """
    Кодирование категориальных признаков
    Пример:
        df_encoded = encode_categorical(df, method='auto')
    """
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns

    encoders = {}
    for col in categorical_cols:
        if method == 'auto':
            if df_encoded[col].nunique() <= 10:
                # One-Hot для небольшого числа категорий
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                encoders[col] = 'onehot'
            else:
                # Label Encoding для большого числа
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        elif method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        elif method == 'onehot':
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)

    print(f"Encoded {len(categorical_cols)} categorical columns")
    return df_encoded, encoders


def scale_features(df, target_col=None, method='standard'):
    """
    Масштабирование признаков
    Пример:
        X_scaled, scaler = scale_features(X, method='standard')
    """
    if target_col and target_col in df.columns:
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        feature_cols = df.columns.tolist()

    # Выбираем только числовые
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"Scaled {len(numeric_cols)} features using {method}")
    return df_scaled, scaler


# ==================== 3. Кластеризация ====================

def kmeans_elbow_method(X, max_k=10, random_state=42):
    """
    Метод локтя для выбора количества кластеров K-Means
    Пример:
        optimal_k = kmeans_elbow_method(X, max_k=10)
    """
    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    # Автоматический выбор (первая точка, где изменение резко замедляется)
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        optimal_k = np.argmin(diffs2) + 2
        print(f"Suggested optimal k: {optimal_k}")
        return optimal_k
    return 3


def perform_clustering(X, n_clusters=3, eps=0.5, min_samples=5):
    """
    Выполнение кластеризации (K-Means и DBSCAN)
    Пример:
        kmeans_labels, dbscan_labels = perform_clustering(X, n_clusters=3)
    """
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)

    print(f"K-Means: {n_clusters} clusters")
    print(f"DBSCAN: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} clusters, "
          f"Noise points: {sum(dbscan_labels == -1)}")

    return kmeans_labels, dbscan_labels


def plot_clusters_2d(X, labels, title="Clustering Results", x_col=0, y_col=1):
    """
    Визуализация кластеров в 2D
    Пример:
        plot_clusters_2d(X, kmeans_labels, title="K-Means Clustering")
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, x_col], X[:, y_col], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel(f'Feature {x_col}')
    plt.ylabel(f'Feature {y_col}')
    plt.title(title)
    plt.show()


# ==================== 4. Классификация ====================

def train_classification_models(X_train, X_test, y_train, y_test):
    """
    Обучение нескольких моделей классификации и сравнение
    Пример:
        results = train_classification_models(X_train, X_test, y_train, y_test)
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        if auc:
            print(f"  AUC-ROC: {auc:.4f}")
        print()

    return results


def plot_confusion_matrix_custom(y_test, y_pred, title="Confusion Matrix"):
    """
    Построение confusion matrix
    Пример:
        plot_confusion_matrix_custom(y_test, y_pred)
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, title="ROC Curve"):
    """
    Построение ROC-кривой
    Пример:
        plot_roc_curve(y_test, y_pred_proba)
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ==================== 5. Регрессия ====================

def train_regression_models(X_train, X_test, y_train, y_test):
    """
    Обучение моделей регрессии
    Пример:
        results = train_regression_models(X_train, X_test, y_train, y_test)
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

        print(f"{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print()

    return results


def plot_predictions(y_test, y_pred, title="Predictions vs Actual"):
    """
    График реальных vs предсказанных значений
    Пример:
        plot_predictions(y_test, y_pred)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.show()


# ==================== 6. Anomaly Detection ====================

def detect_anomalies_iqr(df, column, multiplier=1.5):
    """
    Обнаружение аномалий методом IQR
    Пример:
        anomalies = detect_anomalies_iqr(df, 'price')
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    normal = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Column: {column}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Anomalies found: {len(anomalies)} ({len(anomalies) / len(df) * 100:.2f}%)")

    return anomalies, normal, (lower_bound, upper_bound)


def detect_anomalies_zscore(df, column, threshold=3):
    """
    Обнаружение аномалий методом Z-score
    Пример:
        anomalies = detect_anomalies_zscore(df, 'price')
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    anomalies = df[z_scores > threshold]
    normal = df[z_scores <= threshold]

    print(f"Column: {column}")
    print(f"Threshold: {threshold} standard deviations")
    print(f"Anomalies found: {len(anomalies)} ({len(anomalies) / len(df) * 100:.2f}%)")

    return anomalies, normal


# ==================== 7. Визуализация (дополнительная) ====================

def plot_parallel_coordinates(df, class_col, cols=None):
    """
    Параллельные координаты для кластеризации
    Пример:
        plot_parallel_coordinates(df, 'cluster')
    """
    from pandas.plotting import parallel_coordinates

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if class_col in cols:
            cols.remove(class_col)

    plot_df = df[cols + [class_col]].copy()

    plt.figure(figsize=(12, 6))
    parallel_coordinates(plot_df, class_col, colormap='viridis')
    plt.title('Parallel Coordinates Plot')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def quick_visualization_report(df, target_col=None):
    """
    Быстрый отчет с визуализациями
    Пример:
        quick_visualization_report(df, target_col='price')
    """
    # 1. Матрица корреляций
    print("1. Correlation Matrix")
    plot_correlation_matrix(df)

    # 2. Распределения числовых признаков
    print("\n2. Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plot_distributions(df, numeric_cols[:min(6, len(numeric_cols))])

    # 3. Анализ целевой переменной (если есть)
    if target_col and target_col in df.columns:
        print(f"\n3. Target Analysis: {target_col}")
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        if df[target_col].dtype in ['int64', 'float64']:
            df[target_col].hist(bins=30, edgecolor='black')
            plt.title(f'Distribution of {target_col}')
            plt.xlabel(target_col)
        else:
            df[target_col].value_counts().plot(kind='bar')
            plt.title(f'Value counts of {target_col}')
            plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        if df[target_col].dtype in ['int64', 'float64']:
            df[target_col].boxplot()
            plt.title(f'Boxplot of {target_col}')

        plt.tight_layout()
        plt.show()


# ==================== 7.5. Feature Engineering ====================

def create_interaction_features(df, feature_pairs=None, max_features=10):
    """
    Создание interaction features (произведения признаков)
    Пример:
        df_new = create_interaction_features(df, [('age', 'income'), ('height', 'weight')])
    """
    df_new = df.copy()
    
    if feature_pairs is None:
        # Автоматический выбор самых коррелирующих с целевой
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            # Берём верхний треугольник
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if corr_matrix.iloc[i, j] > 0.5:
                        pairs.append((numeric_cols[i], numeric_cols[j]))
            feature_pairs = pairs[:max_features]
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            new_col_name = f"{col1}_x_{col2}"
            df_new[new_col_name] = df[col1] * df[col2]
            print(f"Created interaction feature: {new_col_name}")
    
    return df_new


def create_polynomial_features(df, degree=2, max_features=10):
    """
    Создание полиномиальных признаков
    Пример:
        df_new = create_polynomial_features(df, degree=2)
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    
    # Создаём имена новых признаков
    poly_names = poly.get_feature_names_out(numeric_cols)
    
    df_new = df.copy()
    for i, name in enumerate(poly_names):
        if name not in numeric_cols:  # Только новые (не оригинальные)
            df_new[name] = poly_features[:, i]
    
    print(f"Created {len(poly_names) - len(numeric_cols)} polynomial features")
    return df_new


# ==================== 7.6. Feature Selection ====================

def select_features_by_variance(df, threshold=0.01):
    """
    Удаление признаков с низкой дисперсией
    Пример:
        df_selected = select_features_by_variance(df, threshold=0.01)
    """
    from sklearn.feature_selection import VarianceThreshold
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df[numeric_cols])
    
    selected_cols = [numeric_cols[i] for i, keep in enumerate(selector.get_support()) if keep]
    removed_cols = [col for col in numeric_cols if col not in selected_cols]
    
    print(f"Removed {len(removed_cols)} low-variance features: {removed_cols}")
    return df[selected_cols + [col for col in df.columns if col not in numeric_cols]]


def select_features_by_correlation(df, target_col, threshold=0.95):
    """
    Удаление сильно коррелирующих признаков
    Пример:
        df_selected = select_features_by_correlation(df, 'target', threshold=0.95)
    """
    if target_col not in df.columns:
        print("Target column not found")
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    df_selected = df.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    
    return df_selected


# ==================== 7.7. Cross-validation wrapper ====================

def cross_validate_model(model, X, y, cv=5, scoring='auto'):
    """
    Кросс-валидация модели с выводом метрик
    Пример:
        scores = cross_validate_model(rf_model, X, y, cv=5)
    """
    if scoring == 'auto':
        if len(np.unique(y)) == 2:
            scoring = 'roc_auc'
        elif len(np.unique(y)) > 2 and len(np.unique(y)) <= 20:
            scoring = 'f1_weighted'
        else:
            scoring = 'r2'
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"Cross-validation ({scoring}):")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"  Individual folds: {scores}")
    
    return scores
    
# ==================== 8. Генерация примеров данных ====================

def generate_sample_data(dataset_type='regression', n_samples=1000, random_state=42):
    """
    Генерация примеров данных для тестирования
    Пример:
        df = generate_sample_data('classification', n_samples=500)
    """
    np.random.seed(random_state)

    if dataset_type == 'regression':
        X = np.random.randn(n_samples, 5)
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.randn(n_samples) * 0.5
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        df['target'] = y
        print("Generated regression dataset")

    elif dataset_type == 'classification':
        X = np.random.randn(n_samples, 5)
        logits = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
        prob = 1 / (1 + np.exp(-logits))
        y = (prob > 0.5).astype(int)
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        df['target'] = y
        print(f"Generated classification dataset (balance: {y.mean():.2f})")

    elif dataset_type == 'clustering':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=3, random_state=random_state)
        df = pd.DataFrame(X, columns=['x', 'y', 'z'])
        df['true_cluster'] = y
        print("Generated clustering dataset")

    else:
        # Mixed data (with categorical and missing)
        df = pd.DataFrame({
            'numeric1': np.random.randn(n_samples),
            'numeric2': np.random.exponential(2, n_samples),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'category2': np.random.choice(['X', 'Y'], n_samples),
            'target': np.random.randn(n_samples) + np.random.randn(n_samples) * 0.5
        })
        # Добавляем пропуски
        for col in ['numeric1', 'numeric2']:
            missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            df.loc[missing_idx, col] = np.nan
        print("Generated mixed dataset with missing values")

    return df


# ==================== 9. SQL утилиты (для демонстрации) ====================

def demo_sql_queries():
    """
    Демонстрация SQL запросов (для понимания)
    Пример:
        demo_sql_queries()
    """
    queries = {
        # 1. Оконные функции
        "employees_above_avg": """
        WITH dept_stats AS (
            SELECT 
                department_id,
                AVG(salary) as avg_salary,
                MAX(salary) as max_salary
            FROM employee
            GROUP BY department_id
        )
        SELECT e.*
        FROM employee e
        JOIN dept_stats d ON e.department_id = d.department_id
        WHERE e.salary > d.avg_salary 
          AND e.salary < d.max_salary;
        """,

        # 2. Рекурсивный CTE (иерархия)
        "folder_hierarchy": """
        WITH RECURSIVE folder_tree AS (
            SELECT 
                folder_id,
                parent_folder_id,
                name,
                CAST(name AS VARCHAR(1000)) as path
            FROM folder_structure
            WHERE parent_folder_id IS NULL

            UNION ALL

            SELECT 
                f.folder_id,
                f.parent_folder_id,
                f.name,
                CAST(ft.path || '/' || f.name AS VARCHAR(1000))
            FROM folder_structure f
            JOIN folder_tree ft ON f.parent_folder_id = ft.folder_id
        )
        SELECT * FROM folder_tree;
        """,

        # 3. Анализ активных сессий
        "max_concurrent_sessions": """
        WITH session_times AS (
            SELECT 
                session_id,
                start_time as time_point,
                1 as delta
            FROM user_sessions

            UNION ALL

            SELECT 
                session_id,
                end_time as time_point,
                -1 as delta
            FROM user_sessions
        ),
        concurrent_count AS (
            SELECT 
                time_point,
                SUM(delta) OVER (ORDER BY time_point) as concurrent_sessions
            FROM session_times
        )
        SELECT MAX(concurrent_sessions) as max_concurrent
        FROM concurrent_count;
        """
    }

    for name, query in queries.items():
        print(f"\n{'=' * 50}")
        print(f"SQL Query: {name}")
        print(f"{'=' * 50}")
        print(query)

    return queries


# ==================== 10. ML Pipeline полный ====================

def create_ml_pipeline(df, target_col, problem_type='auto'):
    """
    Полный ML пайплайн
    Пример:
        model, metrics, X_test, y_test = create_ml_pipeline(df, 'target')
    """
    print("Starting ML Pipeline...")
    print("=" * 50)

    # 1. Определяем тип задачи
    if problem_type == 'auto':
        if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 10:
            problem_type = 'regression'
        else:
            problem_type = 'classification'

    print(f"Problem type: {problem_type}")

    # 2. Обработка данных
    df_clean = handle_missing_values(df)
    df_encoded, encoders = encode_categorical(df_clean)

    # 3. Разделение на признаки и целевую
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Масштабирование
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = X_test.copy()
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 6. Обучение и оценка
    if problem_type == 'classification':
        results = train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")

        # Визуализация для лучшей модели
        y_pred = best_model.predict(X_test_scaled)
        plot_confusion_matrix_custom(y_test, y_pred, f"Confusion Matrix - {best_model_name}")

        if hasattr(best_model, 'predict_proba') and len(np.unique(y)) == 2:
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            plot_roc_curve(y_test, y_pred_proba, f"ROC Curve - {best_model_name}")

    else:
        results = train_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name} (R²: {results[best_model_name]['r2']:.4f})")

        # Визуализация
        y_pred = best_model.predict(X_test_scaled)
        plot_predictions(y_test, y_pred, f"Predictions - {best_model_name}")

    return best_model, results, X_test_scaled, y_test


# ==================== Примеры использования ====================

if __name__ == "__main__":
    """
    Запустите этот файл для демонстрации всех возможностей
    """
    print("UTILS DEMONSTRATION")

    # Пример 1: Генерация и анализ данных
    print("EXAMPLE 1: EDA on regression dataset")

    df_reg = generate_sample_data('regression', n_samples=500)
    quick_eda(df_reg, target_col='target')
    quick_visualization_report(df_reg, target_col='target')

    # Пример 2: Кластеризация
    print("EXAMPLE 2: Clustering")

    df_clust = generate_sample_data('clustering', n_samples=300)
    X_clust = df_clust[['x', 'y', 'z']].values

    # Elbow method
    optimal_k = kmeans_elbow_method(X_clust, max_k=8)

    # Perform clustering
    kmeans_labels, dbscan_labels = perform_clustering(X_clust, n_clusters=optimal_k)

    # Visualize
    plot_clusters_2d(X_clust, kmeans_labels, "K-Means Clustering", x_col=0, y_col=1)

    # Пример 3: Полный ML пайплайн
    print("EXAMPLE 3: Full ML Pipeline")

    df_ml = generate_sample_data('classification', n_samples=300)
    model, results, X_test, y_test = create_ml_pipeline(df_ml, 'target')

    print("DEMONSTRATION COMPLETE!")
