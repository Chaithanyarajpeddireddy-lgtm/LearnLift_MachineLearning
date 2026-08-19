import base64
import io
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(relative_path, nrows=5000):
    root = os.path.dirname(__file__)
    path = os.path.join(root, relative_path)
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    return df


def get_dataset_summary(df):
    numeric = df.select_dtypes(include=["number"])
    summary = {
        "shape": df.shape,
        "columns": list(df.columns[:8]),
        "missing_values": int(df.isna().sum().sum()),
        "numeric_columns": list(numeric.columns[:6]),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    return summary


def preprocess_data(df):
    df_copy = df.copy()
    notes = []

    missing = df_copy.isna().mean() * 100
    removed = missing[missing > 50].index.tolist()
    if removed:
        df_copy = df_copy.drop(columns=removed)
        notes.append(f"Dropped {len(removed)} columns with >50% missing values.")

    numeric = df_copy.select_dtypes(include=["number"])
    if numeric.empty:
        notes.append("No numeric columns available for preprocessing.")
        return df_copy, notes

    for col in numeric.columns:
        median_value = numeric[col].median()
        if numeric[col].isna().any():
            df_copy[col] = numeric[col].fillna(median_value)
            notes.append(f"Filled missing values in '{col}' with median ({median_value:.2f}).")

    scaler = MinMaxScaler()
    df_copy[numeric.columns] = scaler.fit_transform(df_copy[numeric.columns])
    notes.append("Scaled numeric features to the range [0, 1].")

    return df_copy, notes


def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return image_base64


def generate_plots(df):
    plots = {}
    numeric = df.select_dtypes(include=["number"])[:500]

    if not numeric.empty:
        first_col = numeric.columns[0]
        fig = plt.figure(figsize=(6, 4))
        plt.hist(numeric[first_col].dropna(), bins=20, color="#4f46e5", alpha=0.85)
        plt.title(f"Histogram of {first_col}")
        plt.xlabel(first_col)
        plt.ylabel("Frequency")
        plots["histogram"] = plot_to_base64(fig)

        if len(numeric.columns) >= 2:
            second_col = numeric.columns[1]
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(numeric[first_col], numeric[second_col], c="#0ea5e9", alpha=0.6)
            plt.title(f"Scatter: {first_col} vs {second_col}")
            plt.xlabel(first_col)
            plt.ylabel(second_col)
            plots["scatter"] = plot_to_base64(fig)

        corr = numeric.corr()
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Matrix")
        plots["heatmap"] = plot_to_base64(fig)

    return plots


def generate_eda(df):
    numeric = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(exclude=["number"])
    eda = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(len(categorical.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "top_missing": [(str(col), int(value)) for col, value in df.isna().sum().sort_values(ascending=False).head(5).items() if value > 0],
    }
    return eda


def train_logistic_regression(df):
    numeric = df.select_dtypes(include=["number"]).copy()
    result = {
        "status": "Unable to train classification. Not enough numeric columns.",
        "score": None,
        "accuracy": None,
        "feature": None,
        "target": "price_band",
        "classes": None,
    }
    if "price" not in numeric and "price" in df.columns:
        numeric["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "price" not in numeric or len(numeric.columns) < 2:
        return result

    numeric = numeric.dropna(subset=["price"])
    if len(numeric) < 20 or numeric["price"].nunique() < 2:
        return result

    target = (numeric["price"] >= numeric["price"].median()).astype(int)
    feature_col = next((col for col in ["m2_real", "m2_useful", "ground_size"] if col in numeric.columns), None)
    if feature_col is None:
        feature_col = [col for col in numeric.columns if col != "price"][0]

    feature_values = numeric[[feature_col]].fillna(numeric[feature_col].median())
    X_train, X_test, y_train, y_test = train_test_split(
        feature_values, target, test_size=0.2, random_state=42, stratify=target
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    result.update(
        {
            "status": "Logistic regression trained successfully.",
            "score": float(model.score(X_test, y_test)),
            "accuracy": float(model.score(X_test, y_test)),
            "feature": feature_col,
            "classes": ["Lower-priced", "Higher-priced"],
        }
    )
    return result


def generate_model_plots(df, lr_report, logistic_report):
    plots = {}
    numeric = df.select_dtypes(include=["number"]).copy()
    if logistic_report["feature"] and "price" not in numeric and "price" in df.columns:
        numeric["price"] = pd.to_numeric(df["price"], errors="coerce")
    if numeric.empty:
        return plots

    if lr_report["feature"] and lr_report["target"] and {lr_report["feature"], lr_report["target"]}.issubset(numeric.columns):
        feature = lr_report["feature"]
        target = lr_report["target"]
        linear_data = numeric[[feature, target]].dropna()
        fig = plt.figure(figsize=(7, 4))
        plt.scatter(linear_data[feature], linear_data[target], alpha=0.35, color="#e07a5f", label="Observed")
        coefficient = lr_report["coefficients"]
        intercept = lr_report["intercept"]
        line_x = np.linspace(linear_data[feature].min(), linear_data[feature].max(), 100)
        plt.plot(line_x, coefficient * line_x + intercept, color="#264653", linewidth=2.5, label="Regression line")
        plt.title("Linear Regression Fit")
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.legend()
        plots["linear"] = plot_to_base64(fig)

    if logistic_report["feature"] and "price" in numeric:
        feature = logistic_report["feature"]
        logistic_data = numeric[[feature, "price"]].dropna()
        target = (logistic_data["price"] >= logistic_data["price"].median()).astype(int)
        fig = plt.figure(figsize=(7, 4))
        plt.scatter(logistic_data[feature], target, alpha=0.35, color="#2a9d8f")
        plt.title("Logistic Regression Price Bands")
        plt.xlabel(feature)
        plt.ylabel("Price band (0 = lower, 1 = higher)")
        plt.yticks([0, 1], ["Lower-priced", "Higher-priced"])
        plots["logistic"] = plot_to_base64(fig)

    return plots


def train_linear_regression(df):
    numeric = df.select_dtypes(include=["number"])
    result = {
        "status": "Unable to train regression. Not enough numeric columns.",
        "coefficients": None,
        "intercept": None,
        "score": None,
        "target": None,
        "feature": None,
    }

    if len(numeric.columns) < 2:
        return result

    corr = numeric.corr().abs()
    target_col = corr.sum().sort_values(ascending=False).index[1]
    feature_cols = [c for c in numeric.columns if c != target_col]
    if not feature_cols:
        return result

    feature_col = feature_cols[0]
    X = numeric[[feature_col]]
    y = numeric[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    result.update(
        {
            "status": "Linear regression trained successfully.",
            "coefficients": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "score": float(score),
            "feature": feature_col,
            "target": target_col,
        }
    )
    return result


def optimize_data(df):
    numeric = df.select_dtypes(include=["number"]).copy()
    optimization_info = {
        "status": "No numeric data available for optimization.",
        "rolling_features": None,
        "pca_explained_variance": None,
    }

    if numeric.empty:
        return optimization_info

    rolling_info = {}
    for col in numeric.columns[:2]:
        rolling = numeric[col].rolling(window=5, min_periods=1).mean()
        rolling_info[col] = rolling.round(4).tolist()[:5]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    pca = PCA(n_components=min(3, scaled.shape[1]))
    pca.fit(scaled)
    variance = [float(v) for v in pca.explained_variance_ratio_.round(4)]

    optimization_info.update(
        {
            "status": "Data optimization completed.",
            "rolling_features": rolling_info,
            "pca_explained_variance": variance,
        }
    )
    return optimization_info
