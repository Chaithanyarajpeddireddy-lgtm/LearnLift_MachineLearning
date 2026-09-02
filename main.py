import base64
import glob
import io
import os
import warnings
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "Datasets", "archive")


def get_available_regions():
    """Discover all available dataset files and format human-friendly names."""
    if not os.path.exists(DATA_DIR):
        return [{"id": "spanish_houses.csv", "name": "All Spain (National)"}]

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    regions = []
    
    # Priority for All Spain
    default_file = "spanish_houses.csv"
    if os.path.exists(os.path.join(DATA_DIR, default_file)):
        regions.append({"id": default_file, "name": "🇪🇸 All Spain (National Sample)"})

    for f in sorted(files):
        filename = os.path.basename(f)
        if filename == default_file or filename in ["links.csv", "addinfo.csv", "zones.csv", "rentas_PV.csv", "rentas_espanya.csv"]:
            continue
        # Format name from houses_madrid.csv -> Madrid
        name = filename.replace("houses_", "").replace(".csv", "").replace("_", " ").title()
        regions.append({"id": filename, "name": f"📍 {name}"})

    return regions


@lru_cache(maxsize=16)
def load_data(filename="spanish_houses.csv", nrows=6000):
    """Load dataset with caching for fast response times."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        # Fallback to root or default file
        path = os.path.join(DATA_DIR, "spanish_houses.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    
    # Clean price column if string
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 10000]  # Filter out anomalies / placeholders

    # Ensure key numeric columns are properly typed
    numeric_cols = ["m2_real", "m2_useful", "room_num", "bath_num", "swimming_pool", "garden", "terrace", "lift", "air_conditioner"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_dataset_summary(df):
    """Generate high-level dataset statistics."""
    numeric = df.select_dtypes(include=["number"])
    return {
        "shape": df.shape,
        "columns": list(df.columns[:10]),
        "total_columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "numeric_columns": list(numeric.columns[:8]),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "avg_price": float(df["price"].mean()) if "price" in df.columns else 0.0,
        "median_price": float(df["price"].median()) if "price" in df.columns else 0.0,
    }


def generate_eda(df):
    """Perform exploratory data analysis and calculate quality metrics."""
    numeric = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(exclude=["number"])
    
    top_missing = [
        (str(col), int(value), round(float(value) / len(df) * 100, 1))
        for col, value in df.isna().sum().sort_values(ascending=False).head(6).items()
        if value > 0
    ]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "categorical_columns": int(len(categorical.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": round(float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 2) if df.shape[0] > 0 else 0,
        "top_missing": top_missing,
        "price_min": float(df["price"].min()) if "price" in df.columns else 0,
        "price_max": float(df["price"].max()) if "price" in df.columns else 0,
        "price_mean": float(df["price"].mean()) if "price" in df.columns else 0,
        "price_median": float(df["price"].median()) if "price" in df.columns else 0,
    }


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string safely."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return image_base64


def generate_plots(df):
    """Generate thread-safe plots for EDA."""
    plots = {}
    
    # 1. Price Distribution Histogram
    if "price" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Filter 95th percentile for better visualization
        price_cap = df["price"].quantile(0.95)
        filtered_prices = df[df["price"] <= price_cap]["price"] / 1000
        ax.hist(filtered_prices, bins=25, color="#1f5c5b", edgecolor="#ffffff", alpha=0.85)
        ax.set_title("Property Price Distribution (in k€)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Price (Thousands €)")
        ax.set_ylabel("Number of Listings")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plots["histogram"] = plot_to_base64(fig)

    # 2. Scatter Plot: m2_real vs Price
    if "m2_real" in df.columns and "price" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        valid_scatter = df.dropna(subset=["m2_real", "price"])
        valid_scatter = valid_scatter[(valid_scatter["m2_real"] > 10) & (valid_scatter["m2_real"] < 600)]
        price_cap = valid_scatter["price"].quantile(0.95)
        valid_scatter = valid_scatter[valid_scatter["price"] <= price_cap]
        
        ax.scatter(valid_scatter["m2_real"], valid_scatter["price"] / 1000, color="#d4623b", alpha=0.45, edgecolors="none")
        ax.set_title("Size (m²) vs Property Price", fontsize=11, fontweight="bold")
        ax.set_xlabel("Living Area (m²)")
        ax.set_ylabel("Price (Thousands €)")
        ax.grid(True, linestyle="--", alpha=0.3)
        plots["scatter"] = plot_to_base64(fig)

    # 3. Correlation Heatmap of Primary Real Estate Features
    corr_features = [c for c in ["price", "m2_real", "room_num", "bath_num", "swimming_pool", "garden", "terrace", "lift"] if c in df.columns]
    if len(corr_features) >= 3:
        corr_df = df[corr_features].dropna().corr()
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        cax = ax.imshow(corr_df, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(corr_df.index, fontsize=9)
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold")
        
        # Annotate correlation values
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                val = corr_df.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=8)
        
        plots["heatmap"] = plot_to_base64(fig)

    return plots


def build_ml_pipeline(numeric_features, categorical_features, estimator):
    """Build a leak-free scikit-learn pipeline with preprocessing and estimator."""
    transformers = []
    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def prepare_features(df):
    """Select and clean multi-variable feature matrix for modeling."""
    candidate_numeric = ["m2_real", "room_num", "bath_num", "swimming_pool", "garden", "terrace", "lift", "air_conditioner"]
    numeric_features = [col for col in candidate_numeric if col in df.columns]
    
    candidate_cat = ["house_type", "condition"]
    categorical_features = [col for col in candidate_cat if col in df.columns]

    all_features = numeric_features + categorical_features
    clean_df = df[all_features + ["price"]].dropna(subset=["price"]).copy()
    
    # Filter extreme outliers for robust model training
    clean_df = clean_df[(clean_df["price"] >= 20000) & (clean_df["price"] <= clean_df["price"].quantile(0.99))]
    if "m2_real" in clean_df.columns:
        clean_df = clean_df[(clean_df["m2_real"] >= 15) & (clean_df["m2_real"] <= 1200)]

    X = clean_df[all_features]
    y = clean_df["price"]

    return X, y, numeric_features, categorical_features


def train_regression_models(df):
    """Train multivariable Linear, Ridge, and Lasso regression models with full metrics."""
    X, y, num_feats, cat_feats = prepare_features(df)
    
    if len(X) < 50:
        return {"status": "error", "message": "Not enough data for multivariable regression."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear": ("Multivariate Linear Regression", LinearRegression()),
        "ridge": ("Ridge Regression (L2)", Ridge(alpha=10.0)),
        "lasso": ("Lasso Regression (L1)", Lasso(alpha=500.0, max_iter=5000)),
    }

    results = {}
    trained_pipelines = {}

    for key, (name, estimator) in models.items():
        pipeline = build_ml_pipeline(num_feats, cat_feats, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Extract top feature coefficients
        coef_list = []
        try:
            model_step = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = []
            if num_feats:
                feature_names.extend(num_feats)
            if cat_feats and hasattr(preprocessor.named_transformers_["cat"].named_steps["onehot"], "get_feature_names_out"):
                cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_feats)
                feature_names.extend(list(cat_names))

            if hasattr(model_step, "coef_") and len(model_step.coef_) == len(feature_names):
                for f_name, coef in zip(feature_names, model_step.coef_):
                    # Clean up one-hot names for display
                    clean_name = f_name.replace("house_type_", "Type: ").replace("condition_", "Condition: ")
                    coef_list.append({"feature": clean_name, "impact": float(coef)})
                coef_list = sorted(coef_list, key=lambda x: abs(x["impact"]), reverse=True)[:6]
        except Exception:
            coef_list = []

        results[key] = {
            "name": name,
            "r2": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "top_features": coef_list,
        }
        trained_pipelines[key] = pipeline

    return results, trained_pipelines


def train_classification_models(df):
    """Train multivariable Logistic Regression & Decision Trees for Price Tier classification."""
    X, y, num_feats, cat_feats = prepare_features(df)
    
    if len(X) < 50:
        return {"status": "error", "message": "Not enough data for classification."}

    # Target: 1 for High/Luxury (>= Median), 0 for Budget/Standard (< Median)
    median_price = y.median()
    y_class = (y >= median_price).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    models = {
        "logistic": ("Logistic Regression", LogisticRegression(max_iter=1000)),
        "dt_gini": ("Decision Tree (CART - Gini)", DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)),
        "dt_entropy": ("Decision Tree (ID3/C4.5 - Entropy)", DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)),
    }

    results = {}
    trained_pipelines = {}

    for key, (name, estimator) in models.items():
        pipeline = build_ml_pipeline(num_feats, cat_feats, estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        tree_rules = None
        if "dt_" in key:
            try:
                model_step = pipeline.named_steps["model"]
                preprocessor = pipeline.named_steps["preprocessor"]
                feat_names = list(num_feats)
                if cat_feats:
                    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_feats)
                    feat_names.extend(list(cat_names))
                tree_rules = export_text(model_step, feature_names=feat_names, max_depth=3)
            except Exception:
                tree_rules = None

        results[key] = {
            "name": name,
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1": round(float(f1) * 100, 2),
            "confusion_matrix": {
                "tn": cm[0][0] if len(cm) > 0 else 0,
                "fp": cm[0][1] if len(cm) > 0 else 0,
                "fn": cm[1][0] if len(cm) > 1 else 0,
                "tp": cm[1][1] if len(cm) > 1 else 0,
            },
            "tree_rules": tree_rules,
            "classes": ["Standard / Affordable (< Median)", "Premium / Luxury (≥ Median)"],
            "median_threshold": round(float(median_price), 2),
        }
        trained_pipelines[key] = pipeline

    return results, trained_pipelines


def optimize_data(df):
    """Data optimization module: PCA dimensional decomposition & Rolling window analysis."""
    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if numeric.empty or numeric.shape[1] < 2:
        return {
            "status": "Insufficient numeric features for optimization.",
            "rolling_features": None,
            "pca_explained_variance": None,
        }

    # Rolling window smoothing on top features
    rolling_info = {}
    key_cols = [c for c in ["price", "m2_real", "room_num"] if c in numeric.columns]
    for col in key_cols:
        rolling = numeric[col].dropna().rolling(window=5, min_periods=1).mean().round(2)
        rolling_info[col] = rolling.head(6).tolist()

    # PCA analysis
    clean_numeric = numeric.dropna().head(1000)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_numeric)
    
    from sklearn.decomposition import PCA
    n_components = min(4, scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    
    variance_ratio = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    cumulative_variance = round(float(sum(pca.explained_variance_ratio_)) * 100, 2)

    return {
        "status": "Optimization completed successfully.",
        "rolling_features": rolling_info,
        "pca_components": n_components,
        "pca_explained_variance": variance_ratio,
        "pca_cumulative_variance": cumulative_variance,
    }


def predict_house_price(regression_pipeline, classification_pipeline, input_data):
    """
    Perform live real-time inference on arbitrary user input attributes.
    Returns estimated price and tier category.
    """
    df_input = pd.DataFrame([input_data])
    
    # 1. Regression Price Prediction
    pred_price = float(regression_pipeline.predict(df_input)[0])
    pred_price = max(15000.0, round(pred_price, 2))  # Floor negative / anomalous values

    # 2. Classification Prediction
    pred_class_prob = classification_pipeline.predict_proba(df_input)[0]
    is_luxury = int(classification_pipeline.predict(df_input)[0])
    confidence = float(pred_class_prob[is_luxury] * 100)

    category = "Premium / High-End Market" if is_luxury == 1 else "Standard / Affordable Market"

    return {
        "predicted_price": pred_price,
        "predicted_price_formatted": f"€{pred_price:,.0f}",
        "tier": category,
        "is_luxury": bool(is_luxury),
        "confidence": round(confidence, 1),
    }
