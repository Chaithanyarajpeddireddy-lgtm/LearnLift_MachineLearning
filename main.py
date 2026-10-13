"""
main.py — LearnLift OULAD Interactive Machine Learning & Simulation Engine
Comprehensive Machine Learning suite inspired by the Open University Learning Analytics Dataset (OULAD):
  - 12 Classification Models: Logistic, ID3, C4.5, CART, Random Forest, Extra Trees, AdaBoost, GradBoost, XGBoost, LightGBM, KNN, MLP
  - 12 Regression Models: SLR, MLR, Ridge, Lasso, ElasticNet, DTR, Random Forest, Extra Trees, AdaBoost, GradBoost, XGBoost, KNN
  - Interactive What-If Study Simulator
  - Hyperparameter Tuning Playground with Overfitting Diagnosis
  - Batch Student Roster Risk Analyzer
  - Unsupervised K-Means & PCA
"""

import os
import sys

# ─────────────────────────────────────────────────────────────
# WINDOWS 11 / JOBLIB LOKY CORE COUNT FIX
# ─────────────────────────────────────────────────────────────
_n_cpus = os.cpu_count() or 4
os.environ["LOKY_MAX_CPU_COUNT"] = str(_n_cpus)
os.environ["JOBLIB_CPU_COUNT"] = str(_n_cpus)

for sys_dir in [r"C:\Windows\System32", r"C:\Windows\System32\wbem", r"C:\Windows\System32\WindowsPowerShell\v1.0"]:
    if os.path.exists(sys_dir) and sys_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = sys_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import joblib.externals.loky.backend.context as _loky_context
    _loky_context._count_physical_cores = lambda: (_n_cpus, None)
    _loky_context._count_physical_cores_win32 = lambda: (_n_cpus, None)
except Exception:
    pass

try:
    import joblib.externals.loky as _loky
    _loky.cpu_count = lambda only_physical_cores=False: _n_cpus
except Exception:
    pass

try:
    import joblib
    joblib.cpu_count = lambda only_physical_cores=False: _n_cpus
except Exception:
    pass

import base64
import io
import time
import warnings
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Datasets", "Datasets")
if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "studentInfo.csv")):
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")


def get_available_modules():
    """Discover available course modules in OULAD."""
    info_path = os.path.join(DATA_DIR, "studentInfo.csv")
    if not os.path.exists(info_path):
        return [{"id": "ALL", "name": "🎓 All Open University Modules Combined"}]

    try:
        df_info = pd.read_csv(info_path, usecols=["code_module"], nrows=10000)
        mods = sorted(df_info["code_module"].dropna().unique().tolist())
        
        module_titles = {
            "AAA": "Module AAA (Social Sciences & Humanities)",
            "BBB": "Module BBB (Arts, History & Languages)",
            "CCC": "Module CCC (STEM & Physical Sciences)",
            "DDD": "Module DDD (Design, Technology & Environment)",
            "EEE": "Module EEE (Engineering & Applied Physics)",
            "FFF": "Module FFF (Computing & Information Technology)",
            "GGG": "Module GGG (Education & Social Studies)",
        }
        
        modules = [{"id": "ALL", "name": "🎓 All Open University Modules Combined"}]
        for m in mods:
            title = module_titles.get(m, f"Module {m}")
            modules.append({"id": m, "name": f"📚 {title}"})
        return modules
    except Exception:
        return [{"id": "ALL", "name": "🎓 All Open University Modules Combined"}]


@lru_cache(maxsize=16)
def load_and_merge_oulad(code_module="ALL"):
    """
    Load and join studentInfo, studentAssessment, assessments, studentRegistration, and courses.
    Creates a rich, leak-free feature matrix for machine learning.
    """
    info_path = os.path.join(DATA_DIR, "studentInfo.csv")
    assess_path = os.path.join(DATA_DIR, "assessments.csv")
    student_assess_path = os.path.join(DATA_DIR, "studentAssessment.csv")
    reg_path = os.path.join(DATA_DIR, "studentRegistration.csv")

    df_info = pd.read_csv(info_path, low_memory=False)
    
    if code_module and code_module != "ALL":
        df_info = df_info[df_info["code_module"] == code_module].copy()

    df_info.replace("?", np.nan, inplace=True)

    # 1. Aggregate Assessment Data
    if os.path.exists(student_assess_path) and os.path.exists(assess_path):
        df_sa = pd.read_csv(student_assess_path, low_memory=False)
        df_sa.replace("?", np.nan, inplace=True)
        df_sa["score"] = pd.to_numeric(df_sa["score"], errors="coerce")
        df_sa["date_submitted"] = pd.to_numeric(df_sa["date_submitted"], errors="coerce")

        df_assess = pd.read_csv(assess_path, low_memory=False)
        df_assess.replace("?", np.nan, inplace=True)
        df_assess["weight"] = pd.to_numeric(df_assess["weight"], errors="coerce").fillna(0)
        df_assess["date"] = pd.to_numeric(df_assess["date"], errors="coerce")

        df_sa_merged = pd.merge(df_sa, df_assess[["id_assessment", "weight", "date", "assessment_type"]], on="id_assessment", how="left")
        df_sa_merged["submission_lag"] = df_sa_merged["date_submitted"] - df_sa_merged["date"]

        student_perf = df_sa_merged.groupby("id_student").agg(
            avg_assessment_score=("score", "mean"),
            assessments_submitted=("score", "count"),
            avg_submission_lag=("submission_lag", "mean"),
        ).reset_index()

        df_merged = pd.merge(df_info, student_perf, on="id_student", how="left")
    else:
        df_merged = df_info.copy()
        df_merged["avg_assessment_score"] = np.nan
        df_merged["assessments_submitted"] = 0
        df_merged["avg_submission_lag"] = 0

    # 2. Merge Registration Data
    if os.path.exists(reg_path):
        df_reg = pd.read_csv(reg_path, low_memory=False)
        df_reg.replace("?", np.nan, inplace=True)
        df_reg["date_registration"] = pd.to_numeric(df_reg["date_registration"], errors="coerce")
        df_reg["registration_lead_time"] = df_reg["date_registration"].abs().fillna(30)
        
        df_merged = pd.merge(
            df_merged,
            df_reg[["code_module", "code_presentation", "id_student", "registration_lead_time"]],
            on=["code_module", "code_presentation", "id_student"],
            how="left"
        )
    else:
        df_merged["registration_lead_time"] = 30

    # 3. Clean and Cast Numerical Features
    num_cols = ["studied_credits", "num_of_prev_attempts", "avg_assessment_score", "assessments_submitted", "avg_submission_lag", "registration_lead_time"]
    for col in num_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    if "imd_band" in df_merged.columns:
        df_merged["imd_band"] = df_merged["imd_band"].fillna("Missing / Unknown")

    # 4. Target Variables
    df_merged["target_pass"] = df_merged["final_result"].apply(
        lambda x: 1 if str(x).strip() in ["Pass", "Distinction"] else 0
    )

    overall_mean_score = df_merged["avg_assessment_score"].dropna().mean()
    if pd.isna(overall_mean_score):
        overall_mean_score = 65.0
    df_merged["avg_assessment_score"] = df_merged["avg_assessment_score"].fillna(
        df_merged.groupby("final_result")["avg_assessment_score"].transform("mean").fillna(overall_mean_score)
    )

    return df_merged


def get_dataset_info(df):
    """Detailed dataset structure, column data types, missing counts, and preview."""
    cols_info = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        null_count = int(df[col].isna().sum())
        dtype = str(df[col].dtype)
        unique_cnt = int(df[col].nunique())
        sample_val = str(df[col].dropna().iloc[0]) if non_null > 0 else "N/A"
        cols_info.append({
            "name": col,
            "dtype": dtype,
            "non_null": non_null,
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 1),
            "unique_count": unique_cnt,
            "sample": sample_val
        })

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_info": cols_info,
        "head_records": df.head(8).to_dict(orient="records"),
        "tail_records": df.tail(5).to_dict(orient="records"),
        "numeric_summary": df.describe().round(2).to_dict(),
    }


def get_preprocessing_info(df):
    """Preprocessing workflow summary: steps, imputations, scalers, and encoding."""
    num_cols = ["studied_credits", "num_of_prev_attempts", "assessments_submitted", "registration_lead_time"]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = ["code_module", "gender", "highest_education", "imd_band", "age_band", "disability"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    return {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_strategy": {
            "numerical": "Median Imputation via SimpleImputer(strategy='median')",
            "categorical": "Most Frequent / Missing Category Imputation",
        },
        "scaling_strategy": "StandardScaler (Zero mean, unit variance) & MinMaxScaler [0, 1]",
        "encoding_strategy": "OneHotEncoder (handle_unknown='ignore', sparse_output=False)",
        "train_test_split": "80% Training Set, 20% Test Set (Stratified by Academic Outcome)",
        "sample_scaled_preview": df[num_cols].head(5).round(2).to_dict(orient="records") if num_cols else []
    }


def get_preprocessed_dataframe(df):
    """
    Returns cleaned, leak-free DataFrame with imputed values,
    ready for student download as a preprocessed CSV.
    """
    num_cols = ["studied_credits", "num_of_prev_attempts", "assessments_submitted", "registration_lead_time"]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = ["code_module", "gender", "highest_education", "imd_band", "age_band", "disability"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    target_cols = [c for c in ["target_pass", "avg_assessment_score", "final_result"] if c in df.columns]
    cols_to_keep = ([ "id_student" ] if "id_student" in df.columns else []) + num_cols + cat_cols + target_cols
    
    clean_df = df[cols_to_keep].copy()
    for col in num_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    for col in cat_cols:
        clean_df[col] = clean_df[col].fillna("Unknown")
    return clean_df


def get_eda_info(df):
    """Exploratory Data Analysis insights, pass rates, and demographic distributions."""
    outcome_counts = df["final_result"].value_counts().to_dict()
    total = len(df)
    
    edu_stats = []
    if "highest_education" in df.columns:
        for edu, grp in df.groupby("highest_education"):
            if len(grp) > 10:
                p_rate = round(float(grp["target_pass"].mean() * 100), 1)
                avg_sc = round(float(grp["avg_assessment_score"].mean()), 1)
                edu_stats.append({"education": str(edu), "count": len(grp), "pass_rate": p_rate, "avg_score": avg_sc})
        edu_stats = sorted(edu_stats, key=lambda x: x["pass_rate"], reverse=True)

    age_stats = []
    if "age_band" in df.columns:
        for age, grp in df.groupby("age_band"):
            p_rate = round(float(grp["target_pass"].mean() * 100), 1)
            age_stats.append({"age": str(age), "count": len(grp), "pass_rate": p_rate})

    return {
        "total_students": total,
        "pass_rate": round(float(df["target_pass"].mean() * 100), 1),
        "mean_score": round(float(df["avg_assessment_score"].mean()), 1),
        "median_score": round(float(df["avg_assessment_score"].median()), 1),
        "outcome_counts": outcome_counts,
        "edu_stats": edu_stats,
        "age_stats": age_stats,
    }


def plot_to_base64(fig):
    """Convert Matplotlib figure to base64 string safely."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def generate_all_plots(df):
    """Generate high-resolution academic charts."""
    plots = {}
    
    # 1. Assessment Grade Distribution
    if "avg_assessment_score" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        scores = df["avg_assessment_score"].dropna()
        ax.hist(scores, bins=25, color="#2563eb", edgecolor="#ffffff", alpha=0.85)
        ax.axvline(scores.median(), color="#ea580c", linestyle="--", linewidth=2, label=f"Median: {scores.median():.1f}%")
        ax.set_title("Student Assessment Grade Distribution (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Average Grade Score (%)")
        ax.set_ylabel("Number of Students")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plots["histogram"] = plot_to_base64(fig)

    # 2. Final Academic Outcome Breakdown
    if "final_result" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        counts = df["final_result"].value_counts()
        colors = ["#16a34a", "#3b82f6", "#ef4444", "#f59e0b"]
        ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="white", alpha=0.9)
        ax.set_title("Student Final Course Outcomes", fontsize=11, fontweight="bold")
        ax.set_ylabel("Student Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i, v in enumerate(counts.values):
            ax.text(i, v + len(df)*0.01, f"{v:,}", ha="center", fontweight="bold", fontsize=9)
        plots["outcomes"] = plot_to_base64(fig)

    # 3. Studied Credits vs Assessment Grade Scatter
    if "studied_credits" in df.columns and "avg_assessment_score" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        ax.scatter(sample_df["studied_credits"], sample_df["avg_assessment_score"], c=sample_df["target_pass"], cmap="coolwarm", alpha=0.45)
        ax.set_title("Study Credit Load vs. Assessment Score", fontsize=11, fontweight="bold")
        ax.set_xlabel("Enrolled Study Credits")
        ax.set_ylabel("Average Assessment Score (%)")
        ax.grid(True, linestyle="--", alpha=0.3)
        plots["scatter"] = plot_to_base64(fig)

    # 4. Correlation Heatmap
    corr_cols = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time", "target_pass"] if c in df.columns]
    if len(corr_cols) >= 3:
        corr_df = df[corr_cols].dropna().corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.imshow(corr_df, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        clean_labels = [c.replace("avg_assessment_score", "Score").replace("assessments_submitted", "Assessments").replace("studied_credits", "Credits").replace("num_of_prev_attempts", "Attempts").replace("registration_lead_time", "RegDays").replace("target_pass", "PassRate") for c in corr_df.columns]
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(clean_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(clean_labels, fontsize=9)
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold")
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                val = corr_df.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.45 else "black", fontsize=8)
        plots["heatmap"] = plot_to_base64(fig)

    return plots


def prepare_features(df):
    """Extract and prepare leak-free feature matrix for OULAD models."""
    num_features = ["studied_credits", "num_of_prev_attempts", "assessments_submitted", "registration_lead_time"]
    num_features = [c for c in num_features if c in df.columns]

    cat_features = ["code_module", "gender", "highest_education", "imd_band", "age_band", "disability"]
    cat_features = [c for c in cat_features if c in df.columns]

    all_features = num_features + cat_features
    clean_df = df.dropna(subset=["target_pass", "avg_assessment_score"]).copy()

    X = clean_df[all_features]
    y_cls = clean_df["target_pass"].values.astype(int)
    y_reg = clean_df["avg_assessment_score"].values.astype(float)

    return X, y_cls, y_reg, num_features, cat_features


def build_pipeline(num_features, cat_features, estimator):
    """Construct leak-free pipeline with median/mode imputation, scaling, and one-hot encoding."""
    transformers = []
    if num_features:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_features))

    if cat_features:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


# ─────────────────────────────────────────────────────────────
# 1. EXPANDED SUPERVISED CLASSIFICATION SUITE (12 Models)
# ─────────────────────────────────────────────────────────────

CLASSIFICATION_REGISTRY = {
    "lr": {
        "name": "Logistic Regression",
        "desc": "Linear classification model optimizing log-loss with L2 regularization.",
        "type": "Linear Classifier",
        "constructor": lambda: LogisticRegression(max_iter=1000, random_state=42)
    },
    "id3": {
        "name": "ID3 Decision Tree (Entropy, Full Depth)",
        "desc": "Classical Decision Tree utilizing Information Gain (Entropy criterion) without depth limits.",
        "type": "Tree-based",
        "constructor": lambda: DecisionTreeClassifier(criterion="entropy", random_state=42)
    },
    "c45": {
        "name": "C4.5 Decision Tree (Entropy, Depth=10)",
        "desc": "Entropy-based Decision Tree with tree pruning via depth limit (max_depth=10).",
        "type": "Tree-based",
        "constructor": lambda: DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
    },
    "cart": {
        "name": "CART Decision Tree (Gini Impurity, Depth=10)",
        "desc": "Classification and Regression Trees algorithm using Gini Impurity split metric.",
        "type": "Tree-based",
        "constructor": lambda: DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
    },
    "rf": {
        "name": "Random Forest Classifier (Bagging)",
        "desc": "Ensemble of 100 decorrelated decision trees built on bootstrap samples to reduce variance.",
        "type": "Ensemble (Bagging)",
        "constructor": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    },
    "extra_trees": {
        "name": "Extra Trees Classifier (Extremely Randomized)",
        "desc": "Randomized decision trees testing random thresholds for faster training and smoother boundaries.",
        "type": "Ensemble (Bagging)",
        "constructor": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    },
    "adaboost": {
        "name": "AdaBoost Classifier (50 Stumps)",
        "desc": "Adaptive Boosting sequentially focusing on hard-to-classify students using decision stumps.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: AdaBoostClassifier(n_estimators=50, random_state=42)
    },
    "gradboost": {
        "name": "Gradient Boosting Classifier (100 Trees)",
        "desc": "Gradient-boosted decision trees minimizing deviance loss with learning_rate=0.1.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    },
    "xgboost": {
        "name": "XGBoost Classifier",
        "desc": "Extreme Gradient Boosting implementation with histogram binning and second-order loss optimization.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: (
            XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric="logloss")
            if HAS_XGB else HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
        )
    },
    "lightgbm": {
        "name": "LightGBM Classifier",
        "desc": "Light Gradient Boosting Machine utilizing Gradient-based One-Side Sampling (GOSS).",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: (
            LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=-1)
            if HAS_LGBM else HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
        )
    },
    "knn": {
        "name": "K-Nearest Neighbors (KNN, k=7)",
        "desc": "Instance-based non-parametric classifier predicting outcome from 7 closest student profiles.",
        "type": "Instance-Based",
        "constructor": lambda: KNeighborsClassifier(n_neighbors=7)
    },
    "mlp": {
        "name": "Multi-Layer Perceptron (MLP Neural Network)",
        "desc": "Feed-forward artificial neural network with hidden layers (64, 32) and ReLU activations.",
        "type": "Neural Network",
        "constructor": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }
}


MODEL_CACHE = {}


def run_single_classification_model(df, model_key="lr", test_size=0.2, force_retrain=False):
    """Train and evaluate a single requested classification model on demand with in-memory caching."""
    if model_key not in CLASSIFICATION_REGISTRY:
        model_key = "lr"

    cache_key = ("cls", len(df), model_key, round(float(test_size), 2))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    info = CLASSIFICATION_REGISTRY[model_key]
    X, y_cls, _, num_feats, cat_feats = prepare_features(df)
    
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_cls, test_size=test_size, random_state=42, stratify=y_cls
    )

    estimator = info["constructor"]()
    pipeline = build_pipeline(num_feats, cat_feats, estimator)

    t0 = time.time()
    pipeline.fit(Xc_tr, yc_tr)
    fit_time = round((time.time() - t0) * 1000, 2)

    y_pred = pipeline.predict(Xc_te)
    try:
        y_prob = pipeline.predict_proba(Xc_te)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(yc_te, y_pred) * 100
    prec = precision_score(yc_te, y_pred, zero_division=0) * 100
    rec = recall_score(yc_te, y_pred, zero_division=0) * 100
    f1 = f1_score(yc_te, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(yc_te, y_prob) * 100 if y_prob is not None else float("nan")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(yc_te, y_pred)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    spec = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0

    tree_rules = None
    if model_key in ["id3", "c45", "cart"]:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            model_step = pipeline.named_steps["model"]
            feature_names = list(num_feats)
            if cat_feats:
                cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_feats)
                feature_names.extend(list(cat_names))
            tree_rules = export_text(model_step, feature_names=feature_names, max_depth=3)
        except Exception:
            tree_rules = None

    res = {
        "key": model_key,
        "name": info["name"],
        "desc": info["desc"],
        "type": info["type"],
        "fit_time_ms": fit_time,
        "train_samples": len(Xc_tr),
        "test_samples": len(Xc_te),
        "acc": round(acc, 2),
        "prec": round(prec, 2),
        "rec": round(rec, 2),
        "f1": round(f1, 2),
        "spec": round(spec, 2),
        "auc": round(auc, 2) if not np.isnan(auc) else "--",
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tree_rules": tree_rules,
        "pipeline": pipeline
    }
    MODEL_CACHE[cache_key] = res
    return res


# ─────────────────────────────────────────────────────────────
# 2. EXPANDED SUPERVISED REGRESSION SUITE (12 Models)
# ─────────────────────────────────────────────────────────────

REGRESSION_REGISTRY = {
    "slr": {
        "name": "Simple Linear Regression",
        "desc": "Univariate regression mapping the primary engagement feature directly to assessment grade.",
        "type": "Linear Model",
        "constructor": lambda: LinearRegression()
    },
    "mlr": {
        "name": "Multiple Linear Regression",
        "desc": "Multivariate Ordinary Least Squares (OLS) regression utilizing all student demographic & study features.",
        "type": "Linear Model",
        "constructor": lambda: LinearRegression()
    },
    "ridge": {
        "name": "Ridge Regression (L2 Regularization)",
        "desc": "Tikhonov L2 penalty regression (alpha=1.0) preventing feature coefficient inflation.",
        "type": "Regularized Linear",
        "constructor": lambda: Ridge(alpha=1.0, random_state=42)
    },
    "lasso": {
        "name": "Lasso Regression (L1 Regularization)",
        "desc": "Least Absolute Shrinkage Operator with L1 penalty (alpha=0.01) encouraging sparse weights.",
        "type": "Regularized Linear",
        "constructor": lambda: Lasso(alpha=0.01, random_state=42, max_iter=2000)
    },
    "elasticnet": {
        "name": "ElasticNet Regression (L1 + L2)",
        "desc": "Hybrid regularization combining L1 sparsity with L2 coefficient shrinkage (l1_ratio=0.5).",
        "type": "Regularized Linear",
        "constructor": lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42)
    },
    "dtr": {
        "name": "Decision Tree Regressor (Depth=6)",
        "desc": "Non-linear tree regression segmenting student feature space via squared error reduction.",
        "type": "Tree-based",
        "constructor": lambda: DecisionTreeRegressor(criterion="squared_error", max_depth=6, random_state=42)
    },
    "rf_reg": {
        "name": "Random Forest Regressor (Bagging)",
        "desc": "Ensemble of 100 regression trees voting on continuous student assessment grades.",
        "type": "Ensemble (Bagging)",
        "constructor": lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    },
    "extra_trees_reg": {
        "name": "Extra Trees Regressor",
        "desc": "Extremely randomized regression ensemble with high variance reduction.",
        "type": "Ensemble (Bagging)",
        "constructor": lambda: ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    },
    "ada_reg": {
        "name": "AdaBoost Regressor (50 Estimators)",
        "desc": "Sequential adaptive boosting for continuous assessment score estimation.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: AdaBoostRegressor(n_estimators=50, random_state=42)
    },
    "gb_reg": {
        "name": "Gradient Boosting Regressor (100 Trees)",
        "desc": "Additive forward stage-wise gradient boosting minimizing mean squared error loss.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    },
    "xgb_reg": {
        "name": "XGBoost Regressor",
        "desc": "High-performance gradient boosted decision trees for continuous grade regression.",
        "type": "Ensemble (Boosting)",
        "constructor": lambda: (
            XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            if HAS_XGB else HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)
        )
    },
    "knn_reg": {
        "name": "K-Nearest Neighbors Regressor (k=7)",
        "desc": "Non-parametric regression averaging grades of 7 most similar historical students.",
        "type": "Instance-Based",
        "constructor": lambda: KNeighborsRegressor(n_neighbors=7)
    }
}


def run_single_regression_model(df, model_key="mlr", test_size=0.2, force_retrain=False):
    """Train and evaluate a single requested regression model on demand with in-memory caching."""
    if model_key not in REGRESSION_REGISTRY:
        model_key = "mlr"

    cache_key = ("reg", len(df), model_key, round(float(test_size), 2))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    info = REGRESSION_REGISTRY[model_key]
    X, _, y_reg, num_feats, cat_feats = prepare_features(df)

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=test_size, random_state=42
    )

    estimator = info["constructor"]()
    
    if model_key == "slr":
        top_f = [num_feats[0]] if num_feats else [X.columns[0]]
        pipeline = build_pipeline(top_f, [], estimator)
        t0 = time.time()
        pipeline.fit(Xr_tr[top_f], yr_tr)
        fit_time = round((time.time() - t0) * 1000, 2)
        y_pred = pipeline.predict(Xr_te[top_f])
    else:
        pipeline = build_pipeline(num_feats, cat_feats, estimator)
        t0 = time.time()
        pipeline.fit(Xr_tr, yr_tr)
        fit_time = round((time.time() - t0) * 1000, 2)
        y_pred = pipeline.predict(Xr_te)

    r2 = r2_score(yr_te, y_pred)
    mse = mean_squared_error(yr_te, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yr_te, y_pred)

    top_coefficients = []
    if model_key in ["mlr", "ridge", "lasso", "elasticnet"]:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            model_step = pipeline.named_steps["model"]
            feature_names = list(num_feats)
            if cat_feats:
                cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_feats)
                feature_names.extend(list(cat_names))

            for fn, coef in zip(feature_names, model_step.coef_):
                clean_name = fn.replace("highest_education_", "Edu: ").replace("imd_band_", "IMD: ").replace("code_module_", "Module: ").replace("age_band_", "Age: ")
                top_coefficients.append({"feature": clean_name, "impact": float(coef)})
            top_coefficients = sorted(top_coefficients, key=lambda x: abs(x["impact"]), reverse=True)[:8]
        except Exception:
            top_coefficients = []

    res = {
        "key": model_key,
        "name": info["name"],
        "desc": info["desc"],
        "type": info["type"],
        "fit_time_ms": fit_time,
        "train_samples": len(Xr_tr),
        "test_samples": len(Xr_te),
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "top_coefficients": top_coefficients,
        "pipeline": pipeline
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_all_models_summary(df, force_retrain=False):
    """Run lightweight evaluation across all 12 classification and 12 regression models with caching."""
    cache_key = ("all_models_summary", len(df))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    cls_summary = []
    for k in CLASSIFICATION_REGISTRY.keys():
        res = run_single_classification_model(df, k)
        cls_summary.append(res)

    reg_summary = []
    for k in REGRESSION_REGISTRY.keys():
        res = run_single_regression_model(df, k)
        reg_summary.append(res)

    result = (cls_summary, reg_summary)
    MODEL_CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────────
# 3. INTERACTIVE HYPERPARAMETER PLAYGROUND ENGINE
# ─────────────────────────────────────────────────────────────

def train_custom_model(df, model_family, params):
    """
    Train custom model with user-selected hyperparameters from the UI.
    Computes train vs test score to diagnose Overfitting vs Underfitting.
    """
    X, y_cls, y_reg, num_feats, cat_feats = prepare_features(df)
    
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    if model_family == "dt":
        max_depth = int(params.get("max_depth", 5))
        criterion = str(params.get("criterion", "gini"))
        estimator = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    elif model_family == "rf":
        n_est = int(params.get("n_estimators", 50))
        max_depth = int(params.get("max_depth", 8))
        estimator = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42, n_jobs=1)
    elif model_family == "knn":
        n_neighbors = int(params.get("n_neighbors", 5))
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_family == "gb":
        n_est = int(params.get("n_estimators", 50))
        lr = float(params.get("learning_rate", 0.1))
        estimator = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
    else: # lr
        c_val = float(params.get("c_val", 1.0))
        estimator = LogisticRegression(C=c_val, max_iter=1000, random_state=42)

    pipeline = build_pipeline(num_feats, cat_feats, estimator)
    
    t0 = time.time()
    pipeline.fit(Xc_tr, yc_tr)
    fit_time = round((time.time() - t0) * 1000, 2)

    train_acc = accuracy_score(yc_tr, pipeline.predict(Xc_tr)) * 100
    test_acc = accuracy_score(yc_te, pipeline.predict(Xc_te)) * 100
    test_f1 = f1_score(yc_te, pipeline.predict(Xc_te), zero_division=0) * 100

    gap = train_acc - test_acc
    if gap > 12:
        diagnosis = "Overfitting Alert: Model memorizing training data. Increase regularization or reduce tree depth."
        diag_class = "danger"
    elif train_acc < 60:
        diagnosis = "Underfitting Alert: Model too simple to capture patterns. Increase estimators or tree depth."
        diag_class = "warning"
    else:
        diagnosis = "Optimal Generalization: Healthy balance between training and test performance."
        diag_class = "success"

    cm = confusion_matrix(yc_te, pipeline.predict(Xc_te))

    return {
        "train_acc": round(train_acc, 2),
        "test_acc": round(test_acc, 2),
        "test_f1": round(test_f1, 2),
        "gap": round(gap, 2),
        "fit_time_ms": fit_time,
        "diagnosis": diagnosis,
        "diag_class": diag_class,
        "confusion_matrix": {
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1])
        }
    }


# ─────────────────────────────────────────────────────────────
# 4. INTERACTIVE WHAT-IF STUDY SIMULATOR
# ─────────────────────────────────────────────────────────────

def simulate_study_impact(reg_pipeline, clf_pipeline, base_input, modified_input):
    """
    Computes What-If impact showing delta changes in predicted grade and pass probability.
    """
    res_base = predict_student_outcome(reg_pipeline, clf_pipeline, base_input)
    res_mod = predict_student_outcome(reg_pipeline, clf_pipeline, modified_input)

    delta_grade = round(res_mod["predicted_grade"] - res_base["predicted_grade"], 1)
    delta_pass_prob = round(res_mod["pass_probability"] - res_base["pass_probability"], 1)

    return {
        "base": res_base,
        "simulated": res_mod,
        "delta_grade": delta_grade,
        "delta_pass_prob": delta_pass_prob,
        "improved": delta_grade >= 0,
    }


# ─────────────────────────────────────────────────────────────
# 5. BATCH STUDENT ROSTER EVALUATOR
# ─────────────────────────────────────────────────────────────

def predict_student_batch(reg_pipeline, clf_pipeline, student_records):
    """
    Evaluates a batch cohort of student profiles, partitioning into performance tiers.
    """
    df_batch = pd.DataFrame(student_records)
    preds_grade = reg_pipeline.predict(df_batch)
    probs_pass = clf_pipeline.predict_proba(df_batch)[:, 1]

    results = []
    for i, row in df_batch.iterrows():
        grade = round(max(0.0, min(100.0, float(preds_grade[i]))), 1)
        p_prob = round(float(probs_pass[i]) * 100, 1)

        if p_prob >= 75:
            tier = "Low Risk (Dean's List / Pass)"
            badge = "success"
        elif p_prob >= 50:
            tier = "Moderate Risk (Passing)"
            badge = "warning"
        else:
            tier = "High Risk (Urgent Intervention)"
            badge = "danger"

        results.append({
            "student_id": row.get("id_student", f"STD-{1000 + i}"),
            "module": row.get("code_module", "BBB"),
            "education": row.get("highest_education", "A Level"),
            "credits": row.get("studied_credits", 60),
            "submitted": row.get("assessments_submitted", 4),
            "predicted_grade": f"{grade:.1f}%",
            "pass_prob": f"{p_prob:.1f}%",
            "tier": tier,
            "badge": badge
        })

    return results


# ─────────────────────────────────────────────────────────────
# 6. EXPANDED UNSUPERVISED LEARNING SUITE (K-Means, Hierarchical, DBSCAN, PCA, t-SNE, UMAP)
# ─────────────────────────────────────────────────────────────

def run_unsupervised_kmeans(df, n_clusters=3, force_retrain=False):
    cache_key = ("kmeans", len(df), n_clusters)
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    cluster_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_data = df[cluster_features].dropna()
    if len(clean_data) > 3000:
        clean_data = clean_data.sample(3000, random_state=42)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    k_range = [2, 3, 4, 5, 6]
    elbow_inertias = []
    silhouette_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_data)
        elbow_inertias.append(round(float(km.inertia_), 1))
        sil = silhouette_score(scaled_data, km.labels_)
        silhouette_scores.append(round(float(sil), 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    t0 = time.time()
    cluster_labels = kmeans.fit_predict(scaled_data)
    fit_time = round((time.time() - t0) * 1000, 2)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(scaled_data)
    centers_2d = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster_labels, cmap="Set1", alpha=0.6, edgecolors="none", s=25)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=180, label="Cluster Centroids", edgecolors="white", linewidth=1.5)
    ax.set_title(f"K-Means Student Clusters (k={n_clusters}) in 2D PCA Space", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    cluster_plot_b64 = plot_to_base64(fig)

    clean_data_copy = clean_data.copy()
    clean_data_copy["Cluster"] = cluster_labels
    profiles = []
    for c_id in range(n_clusters):
        c_grp = clean_data_copy[clean_data_copy["Cluster"] == c_id]
        profiles.append({
            "cluster_id": c_id,
            "student_count": len(c_grp),
            "pct_of_total": round(len(c_grp) / len(clean_data) * 100, 1),
            "avg_score": round(float(c_grp["avg_assessment_score"].mean()), 1) if "avg_assessment_score" in c_grp else 0,
            "avg_assessments": round(float(c_grp["assessments_submitted"].mean()), 1) if "assessments_submitted" in c_grp else 0,
            "avg_credits": round(float(c_grp["studied_credits"].mean()), 1) if "studied_credits" in c_grp else 0,
            "avg_attempts": round(float(c_grp["num_of_prev_attempts"].mean()), 2) if "num_of_prev_attempts" in c_grp else 0,
        })

    res = {
        "k": n_clusters,
        "k_range": k_range,
        "elbow_inertias": elbow_inertias,
        "silhouette_scores": silhouette_scores,
        "cluster_plot": cluster_plot_b64,
        "profiles": profiles,
        "fit_time_ms": fit_time,
        "features_used": cluster_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_unsupervised_hierarchical(df, n_clusters=3, linkage_method="ward", force_retrain=False):
    """
    Hierarchical (Agglomerative) Clustering with Dendrogram tree visualization.
    """
    cache_key = ("hierarchical", len(df), n_clusters, linkage_method)
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    cluster_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_data = df[cluster_features].dropna()
    if len(clean_data) > 2000:
        clean_data = clean_data.sample(2000, random_state=42)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    if linkage_method not in ["ward", "complete", "average"]:
        linkage_method = "ward"

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    t0 = time.time()
    cluster_labels = agg.fit_predict(scaled_data)
    fit_time = round((time.time() - t0) * 1000, 2)

    sil = round(float(silhouette_score(scaled_data, cluster_labels)), 3)

    # 1. Dendrogram Visualization
    dendro_plot_b64 = None
    if HAS_SCIPY:
        try:
            dendro_sample = scaled_data[:40]
            Z = linkage(dendro_sample, method=linkage_method)
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            dendrogram(Z, ax=ax, truncate_mode=None, p=12, color_threshold=Z[-(n_clusters-1), 2] if n_clusters > 1 else 0)
            ax.set_title(f"Hierarchical Tree Dendrogram ({linkage_method.capitalize()} Linkage)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Sampled Student Leaf Node Index")
            ax.set_ylabel("Euclidean Cluster Dissimilarity")
            ax.grid(True, linestyle="--", alpha=0.3)
            dendro_plot_b64 = plot_to_base64(fig)
        except Exception:
            dendro_plot_b64 = None

    # 2. 2D Cluster Projection
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster_labels, cmap="Accent", alpha=0.65, s=28)
    ax.set_title(f"Hierarchical Student Personas (k={n_clusters}) in 2D PCA Space", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.grid(True, linestyle="--", alpha=0.3)
    cluster_plot_b64 = plot_to_base64(fig)

    clean_data_copy = clean_data.copy()
    clean_data_copy["Cluster"] = cluster_labels
    profiles = []
    for c_id in range(n_clusters):
        c_grp = clean_data_copy[clean_data_copy["Cluster"] == c_id]
        profiles.append({
            "cluster_id": c_id,
            "student_count": len(c_grp),
            "pct_of_total": round(len(c_grp) / len(clean_data) * 100, 1),
            "avg_score": round(float(c_grp["avg_assessment_score"].mean()), 1) if "avg_assessment_score" in c_grp else 0,
            "avg_assessments": round(float(c_grp["assessments_submitted"].mean()), 1) if "assessments_submitted" in c_grp else 0,
            "avg_credits": round(float(c_grp["studied_credits"].mean()), 1) if "studied_credits" in c_grp else 0,
            "avg_attempts": round(float(c_grp["num_of_prev_attempts"].mean()), 2) if "num_of_prev_attempts" in c_grp else 0,
        })

    res = {
        "k": n_clusters,
        "linkage": linkage_method,
        "silhouette_score": sil,
        "fit_time_ms": fit_time,
        "dendrogram_plot": dendro_plot_b64,
        "cluster_plot": cluster_plot_b64,
        "profiles": profiles,
        "features_used": cluster_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_unsupervised_dbscan(df, eps=0.6, min_samples=5, force_retrain=False):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    Identifies high-density student clusters and flags isolated anomaly/noise students.
    """
    cache_key = ("dbscan", len(df), float(eps), int(min_samples))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    cluster_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_data = df[cluster_features].dropna()
    if len(clean_data) > 2500:
        clean_data = clean_data.sample(2500, random_state=42)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    t0 = time.time()
    labels = db.fit_predict(scaled_data)
    fit_time = round((time.time() - t0) * 1000, 2)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(list(labels).count(-1))
    noise_pct = round(n_noise / len(labels) * 100, 1)

    if n_clusters > 1 and len(scaled_data[labels != -1]) > 10:
        sil = round(float(silhouette_score(scaled_data[labels != -1], labels[labels != -1])), 3)
    else:
        sil = "--"

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    mask_normal = labels != -1
    if np.any(mask_normal):
        ax.scatter(pca_2d[mask_normal, 0], pca_2d[mask_normal, 1], c=labels[mask_normal], cmap="tab10", alpha=0.65, s=26, label="Clustered Profiles")
    mask_noise = labels == -1
    if np.any(mask_noise):
        ax.scatter(pca_2d[mask_noise, 0], pca_2d[mask_noise, 1], c="crimson", marker="x", s=45, alpha=0.85, label=f"Anomalies / Noise ({n_noise})")
    
    ax.set_title(f"DBSCAN Density Clustering (eps={eps}, min_samples={min_samples})", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    dbscan_plot_b64 = plot_to_base64(fig)

    clean_data_copy = clean_data.copy()
    clean_data_copy["Cluster"] = labels
    
    # Anomaly comparison breakdown
    noise_grp = clean_data_copy[clean_data_copy["Cluster"] == -1]
    normal_grp = clean_data_copy[clean_data_copy["Cluster"] != -1]

    anomaly_stats = {
        "noise_count": n_noise,
        "noise_pct": noise_pct,
        "noise_avg_score": round(float(noise_grp["avg_assessment_score"].mean()), 1) if len(noise_grp) else 0,
        "normal_avg_score": round(float(normal_grp["avg_assessment_score"].mean()), 1) if len(normal_grp) else 0,
        "noise_avg_credits": round(float(noise_grp["studied_credits"].mean()), 1) if len(noise_grp) else 0,
        "normal_avg_credits": round(float(normal_grp["studied_credits"].mean()), 1) if len(normal_grp) else 0,
        "noise_avg_assessments": round(float(noise_grp["assessments_submitted"].mean()), 1) if len(noise_grp) else 0,
        "normal_avg_assessments": round(float(normal_grp["assessments_submitted"].mean()), 1) if len(normal_grp) else 0,
    }

    res = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "silhouette_score": sil,
        "fit_time_ms": fit_time,
        "dbscan_plot": dbscan_plot_b64,
        "anomaly_stats": anomaly_stats,
        "features_used": cluster_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_unsupervised_tsne(df, perplexity=30, n_iter=1000, force_retrain=False):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE)
    Non-linear manifold decomposition preserving local student neighborhoods.
    """
    cache_key = ("tsne", len(df), int(perplexity), int(n_iter))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    tsne_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_data = df[tsne_features].dropna()
    if len(clean_data) > 1500:
        clean_data = clean_data.sample(1500, random_state=42)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    perp = min(perplexity, max(5, len(clean_data) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_iter, random_state=42)
    
    t0 = time.time()
    tsne_2d = tsne.fit_transform(scaled_data)
    fit_time = round((time.time() - t0) * 1000, 2)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    target_colors = df.loc[clean_data.index, "target_pass"].values if "target_pass" in df.columns else np.zeros(len(clean_data))
    scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=target_colors, cmap="coolwarm", alpha=0.6, s=26)
    ax.set_title(f"t-SNE 2D Manifold Projection (Perplexity={perp})", fontsize=11, fontweight="bold")
    ax.set_xlabel("t-SNE Embedding Dimension 1")
    ax.set_ylabel("t-SNE Embedding Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["At-Risk / Fail", "Pass / Distinction"])
    tsne_plot_b64 = plot_to_base64(fig)

    res = {
        "perplexity": perp,
        "n_samples": len(clean_data),
        "fit_time_ms": fit_time,
        "tsne_plot": tsne_plot_b64,
        "features": tsne_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_unsupervised_umap(df, n_neighbors=15, min_dist=0.1, force_retrain=False):
    """
    Uniform Manifold Approximation and Projection (UMAP)
    Preserves both local and global topological structure across student learning trajectories.
    """
    cache_key = ("umap", len(df), int(n_neighbors), float(min_dist))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    umap_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_data = df[umap_features].dropna()
    if len(clean_data) > 1800:
        clean_data = clean_data.sample(1800, random_state=42)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)

    is_fallback = False
    t0 = time.time()
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        umap_2d = reducer.fit_transform(scaled_data)
    else:
        is_fallback = True
        reducer = Isomap(n_neighbors=n_neighbors, n_components=2)
        umap_2d = reducer.fit_transform(scaled_data)
    fit_time = round((time.time() - t0) * 1000, 2)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    target_scores = df.loc[clean_data.index, "avg_assessment_score"].values if "avg_assessment_score" in df.columns else np.zeros(len(clean_data))
    scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=target_scores, cmap="viridis", alpha=0.6, s=26)
    
    title_str = f"UMAP 2D Manifold Projection (n_neighbors={n_neighbors})" if not is_fallback else f"Non-Linear Manifold Projection (Isomap/UMAP Fallback, k={n_neighbors})"
    ax.set_title(title_str, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1" if not is_fallback else "Manifold Coordinate 1")
    ax.set_ylabel("UMAP Dimension 2" if not is_fallback else "Manifold Coordinate 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average Assessment Grade (%)")
    umap_plot_b64 = plot_to_base64(fig)

    res = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "is_fallback": is_fallback,
        "n_samples": len(clean_data),
        "fit_time_ms": fit_time,
        "umap_plot": umap_plot_b64,
        "features": umap_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


def run_unsupervised_pca(df, n_components=4, force_retrain=False):
    cache_key = ("pca", len(df), int(n_components))
    if not force_retrain and cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    pca_features = [c for c in ["avg_assessment_score", "assessments_submitted", "studied_credits", "num_of_prev_attempts", "registration_lead_time"] if c in numeric.columns]
    
    clean_num = df[pca_features].dropna()
    if len(clean_num) > 3000:
        clean_num = clean_num.sample(3000, random_state=42)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_num)

    pca = PCA(n_components=min(n_components, scaled.shape[1]), random_state=42)
    t0 = time.time()
    pca_transformed = pca.fit_transform(scaled)
    fit_time = round((time.time() - t0) * 1000, 2)

    variance_ratios = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    cumulative_variance = round(float(sum(pca.explained_variance_ratio_)) * 100, 2)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    target_colors = df.loc[clean_num.index, "target_pass"].values if "target_pass" in df.columns else np.zeros(len(clean_num))
    ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=target_colors, cmap="coolwarm", alpha=0.5, s=25)
    ax.set_title("Student PCA 2D Projection (Blue = At-Risk, Red = Pass)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance_ratios[0]}% Variance)")
    ax.set_ylabel(f"PC2 ({variance_ratios[1]}% Variance)")
    ax.grid(True, linestyle="--", alpha=0.3)
    pca_plot_b64 = plot_to_base64(fig)

    res = {
        "n_components": pca.n_components_,
        "explained_variance": variance_ratios,
        "cumulative_variance": cumulative_variance,
        "fit_time_ms": fit_time,
        "pca_plot": pca_plot_b64,
        "features": pca_features,
    }
    MODEL_CACHE[cache_key] = res
    return res


# ─────────────────────────────────────────────────────────────
# 7. LIVE INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────

def predict_student_outcome(reg_pipeline, clf_pipeline, input_dict):
    df_input = pd.DataFrame([input_dict])

    pred_grade = float(reg_pipeline.predict(df_input)[0])
    pred_grade = max(0.0, min(100.0, round(pred_grade, 1)))

    pred_class_prob = clf_pipeline.predict_proba(df_input)[0]
    is_pass = int(clf_pipeline.predict(df_input)[0])
    pass_prob = float(pred_class_prob[1] * 100)
    fail_prob = float(pred_class_prob[0] * 100)

    if pass_prob >= 75:
        risk_level = "Low Risk (Dean's List / High Pass)"
        tier_class = "success"
        action = "Student is well on track. Maintain current submission pace."
    elif pass_prob >= 50:
        risk_level = "Moderate Risk (Borderline Pass)"
        tier_class = "warning"
        action = "Recommend supplementary study hours and weekly tutor office visits."
    else:
        risk_level = "High Academic Risk (Probable Fail / Dropout)"
        tier_class = "danger"
        action = "Immediate academic intervention required. Reduce concurrent study credits and schedule 1-on-1 advisor review."

    return {
        "predicted_grade": pred_grade,
        "predicted_grade_formatted": f"{pred_grade:.1f}%",
        "is_pass": bool(is_pass),
        "pass_probability": round(pass_prob, 1),
        "dropout_risk_probability": round(fail_prob, 1),
        "risk_level": risk_level,
        "tier_class": tier_class,
        "recommended_action": action,
    }
