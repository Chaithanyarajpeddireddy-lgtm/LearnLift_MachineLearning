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

import io
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from main import (
    CLASSIFICATION_REGISTRY,
    REGRESSION_REGISTRY,
    generate_all_plots,
    get_available_modules,
    get_dataset_info,
    get_eda_info,
    get_preprocessed_dataframe,
    get_preprocessing_info,
    load_and_merge_oulad,
    predict_student_batch,
    predict_student_outcome,
    run_all_models_summary,
    run_single_classification_model,
    run_single_regression_model,
    run_unsupervised_dbscan,
    run_unsupervised_hierarchical,
    run_unsupervised_kmeans,
    run_unsupervised_pca,
    run_unsupervised_tsne,
    run_unsupervised_umap,
    simulate_study_impact,
    train_custom_model,
)

app = Flask(__name__, static_folder="Static", template_folder="templates")

DATA_CACHE = {}


def get_cached_dataset(module_id="ALL"):
    """Retrieve pre-processed DataFrame from memory or load it once."""
    if module_id not in DATA_CACHE:
        try:
            DATA_CACHE[module_id] = load_and_merge_oulad(module_id)
        except Exception as err:
            print(f"Error loading module {module_id}: {err}")
            DATA_CACHE[module_id] = load_and_merge_oulad("ALL")
    return DATA_CACHE[module_id]


@app.context_processor
def inject_global_data():
    """Make course modules and active module available globally across all templates."""
    selected_module = request.args.get("module", "ALL")
    return {
        "available_modules": get_available_modules(),
        "selected_module": selected_module,
    }


# ─────────────────────────────────────────────────────────────
# 1. OVERVIEW & DATA PIPELINE
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)
    eda = get_eda_info(df)
    dataset_info = get_dataset_info(df)
    return render_template("index.html", eda=eda, dataset_info=dataset_info)


@app.route("/dataset")
def dataset_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)
    dataset_info = get_dataset_info(df)
    return render_template("dataset.html", dataset_info=dataset_info)


@app.route("/preprocessing")
def preprocessing_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)
    prep_info = get_preprocessing_info(df)
    return render_template("preprocessing.html", prep_info=prep_info)


@app.route("/eda")
def eda_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)
    eda = get_eda_info(df)
    return render_template("eda.html", eda=eda)


@app.route("/visualization")
def visualization_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)
    plots = generate_all_plots(df)
    return render_template("visualization.html", plots=plots)


# ─────────────────────────────────────────────────────────────
# 2. SUPERVISED LEARNING (24 Models)
# ─────────────────────────────────────────────────────────────
@app.route("/supervised")
@app.route("/models")
def supervised_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)

    model_type = request.args.get("type", "classification")
    selected_model = request.args.get("model", "lr" if model_type == "classification" else "mlr")
    should_run = request.args.get("run", "0") == "1"

    model_result = None
    if should_run:
        if model_type == "classification":
            if selected_model not in CLASSIFICATION_REGISTRY:
                selected_model = "lr"
            model_result = run_single_classification_model(df, selected_model)
        else:
            if selected_model not in REGRESSION_REGISTRY:
                selected_model = "mlr"
            model_result = run_single_regression_model(df, selected_model)

    return render_template(
        "models.html",
        model_type=model_type,
        selected_model=selected_model,
        model_result=model_result,
        classification_models=CLASSIFICATION_REGISTRY,
        regression_models=REGRESSION_REGISTRY,
    )


@app.route("/api/run-supervised", methods=["POST"])
def api_run_supervised():
    """Dynamically train and evaluate a single requested supervised algorithm."""
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        module_id = data.get("module", "ALL")
        df = get_cached_dataset(module_id)

        model_type = data.get("type", "classification")
        model_key = data.get("model", "lr" if model_type == "classification" else "mlr")
        test_size = float(data.get("test_size", 0.2))

        if model_type == "classification":
            if model_key not in CLASSIFICATION_REGISTRY:
                model_key = "lr"
            res = run_single_classification_model(df, model_key, test_size=test_size, force_retrain=True)
        else:
            if model_key not in REGRESSION_REGISTRY:
                model_key = "mlr"
            res = run_single_regression_model(df, model_key, test_size=test_size, force_retrain=True)

        clean_res = {k: v for k, v in res.items() if k != "pipeline"}
        return jsonify({"status": "success", "result": clean_res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# 3. INTERACTIVE WHAT-IF STUDY SIMULATOR
# ─────────────────────────────────────────────────────────────
@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Real-time What-If simulation API."""
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        module_id = data.get("code_module", "ALL")
        df = get_cached_dataset(module_id)

        clf_res = run_single_classification_model(df, "lightgbm")
        reg_res = run_single_regression_model(df, "gb_reg")

        base_input = {
            "code_module": str(data.get("code_module", "BBB")),
            "gender": str(data.get("gender", "M")),
            "highest_education": str(data.get("highest_education", "A Level or Equivalent")),
            "imd_band": str(data.get("imd_band", "50-60%")),
            "age_band": str(data.get("age_band", "0-35")),
            "disability": str(data.get("disability", "N")),
            "studied_credits": float(data.get("base_credits", 60)),
            "num_of_prev_attempts": float(data.get("base_attempts", 0)),
            "assessments_submitted": float(data.get("base_submitted", 3)),
            "registration_lead_time": float(data.get("base_lead_time", 20)),
        }

        mod_input = {
            "code_module": str(data.get("code_module", "BBB")),
            "gender": str(data.get("gender", "M")),
            "highest_education": str(data.get("highest_education", "A Level or Equivalent")),
            "imd_band": str(data.get("imd_band", "50-60%")),
            "age_band": str(data.get("age_band", "0-35")),
            "disability": str(data.get("disability", "N")),
            "studied_credits": float(data.get("sim_credits", 60)),
            "num_of_prev_attempts": float(data.get("sim_attempts", 0)),
            "assessments_submitted": float(data.get("sim_submitted", 6)),
            "registration_lead_time": float(data.get("sim_lead_time", 45)),
        }

        sim_res = simulate_study_impact(reg_res["pipeline"], clf_res["pipeline"], base_input, mod_input)
        sim_res["status"] = "success"
        return jsonify(sim_res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING PLAYGROUND
# ─────────────────────────────────────────────────────────────
@app.route("/playground")
def playground_page():
    return render_template("playground.html")


@app.route("/api/tune", methods=["POST"])
def api_tune():
    """Live hyperparameter training API with overfitting diagnosis."""
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        module_id = data.get("module", "ALL")
        df = get_cached_dataset(module_id)

        model_family = data.get("algorithm", "dt")
        tune_res = train_custom_model(df, model_family, data)
        tune_res["status"] = "success"
        return jsonify(tune_res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# 5. BATCH STUDENT ROSTER EVALUATOR
# ─────────────────────────────────────────────────────────────
@app.route("/batch")
def batch_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)

    # Generate sample 12-student cohort from dataset for instant testing
    sample_records = df.sample(min(12, len(df)), random_state=42).to_dict(orient="records")
    return render_template("batch.html", sample_records=sample_records)


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Evaluate batch roster of students."""
    try:
        data = request.get_json(force=True) or {}
        records = data.get("students", [])
        module_id = data.get("module", "ALL")
        df = get_cached_dataset(module_id)

        if not records:
            records = df.sample(min(15, len(df)), random_state=42).to_dict(orient="records")

        clf_res = run_single_classification_model(df, "lightgbm")
        reg_res = run_single_regression_model(df, "gb_reg")

        results = predict_student_batch(reg_res["pipeline"], clf_res["pipeline"], records)
        return jsonify({"status": "success", "results": results, "total": len(results)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/export-preprocessed")
def export_preprocessed():
    """Stream cleaned, preprocessed OULAD dataset as a CSV attachment."""
    try:
        module_id = request.args.get("module", "ALL")
        df = get_cached_dataset(module_id)
        clean_df = get_preprocessed_dataframe(df)
        csv_buffer = io.StringIO()
        clean_df.to_csv(csv_buffer, index=False)
        return Response(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=oulad_preprocessed_{module_id}.csv"}
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# 6. UNSUPERVISED LEARNING & INFERENCE
# ─────────────────────────────────────────────────────────────
@app.route("/unsupervised")
def unsupervised_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)

    tab = request.args.get("tab", "kmeans")
    k = int(request.args.get("k", 3))
    if k not in [2, 3, 4, 5, 6]:
        k = 3

    eps = float(request.args.get("eps", 0.6))
    min_samples = int(request.args.get("min_samples", 5))
    perplexity = int(request.args.get("perplexity", 30))
    linkage_method = request.args.get("linkage", "ward")
    should_run = request.args.get("run", "0") == "1"

    kmeans_res = None
    hierarchical_res = None
    dbscan_res = None
    pca_res = None
    tsne_res = None
    umap_res = None

    if should_run:
        if tab == "kmeans":
            kmeans_res = run_unsupervised_kmeans(df, n_clusters=k)
        elif tab == "hierarchical":
            hierarchical_res = run_unsupervised_hierarchical(df, n_clusters=k, linkage_method=linkage_method)
        elif tab == "dbscan":
            dbscan_res = run_unsupervised_dbscan(df, eps=eps, min_samples=min_samples)
        elif tab == "pca":
            pca_res = run_unsupervised_pca(df, n_components=4)
        elif tab == "tsne":
            tsne_res = run_unsupervised_tsne(df, perplexity=perplexity)
        elif tab == "umap":
            umap_res = run_unsupervised_umap(df)

    return render_template(
        "unsupervised.html",
        active_tab=tab,
        k=k,
        eps=eps,
        min_samples=min_samples,
        perplexity=perplexity,
        linkage_method=linkage_method,
        kmeans_res=kmeans_res,
        hierarchical_res=hierarchical_res,
        dbscan_res=dbscan_res,
        pca_res=pca_res,
        tsne_res=tsne_res,
        umap_res=umap_res,
    )


@app.route("/api/run-unsupervised", methods=["POST"])
def api_run_unsupervised():
    """Dynamically run only the requested unsupervised clustering or manifold algorithm."""
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        module_id = data.get("module", "ALL")
        df = get_cached_dataset(module_id)
        algo = data.get("algorithm", "kmeans")

        if algo == "kmeans":
            k = int(data.get("k", 3))
            if k not in [2, 3, 4, 5, 6]:
                k = 3
            res = run_unsupervised_kmeans(df, n_clusters=k, force_retrain=True)
        elif algo == "hierarchical":
            k = int(data.get("k", 3))
            linkage_m = str(data.get("linkage", "ward"))
            res = run_unsupervised_hierarchical(df, n_clusters=k, linkage_method=linkage_m, force_retrain=True)
        elif algo == "dbscan":
            eps = float(data.get("eps", 0.6))
            min_samples = int(data.get("min_samples", 5))
            res = run_unsupervised_dbscan(df, eps=eps, min_samples=min_samples, force_retrain=True)
        elif algo == "pca":
            n_comp = int(data.get("n_components", 4))
            res = run_unsupervised_pca(df, n_components=n_comp, force_retrain=True)
        elif algo == "tsne":
            perplexity = int(data.get("perplexity", 30))
            n_iter = int(data.get("n_iter", 500))
            res = run_unsupervised_tsne(df, perplexity=perplexity, n_iter=n_iter, force_retrain=True)
        elif algo == "umap":
            n_neighbors = int(data.get("n_neighbors", 15))
            min_dist = float(data.get("min_dist", 0.1))
            res = run_unsupervised_umap(df, n_neighbors=n_neighbors, min_dist=min_dist, force_retrain=True)
        else:
            return jsonify({"status": "error", "message": f"Unknown unsupervised algorithm: {algo}"}), 400

        return jsonify({"status": "success", "algorithm": algo, "result": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Real-time single student inference."""
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        module_id = data.get("code_module", "ALL")
        df = get_cached_dataset(module_id)

        clf_res = run_single_classification_model(df, "lightgbm")
        reg_res = run_single_regression_model(df, "gb_reg")

        input_dict = {
            "code_module": str(data.get("code_module", "BBB")),
            "gender": str(data.get("gender", "M")),
            "highest_education": str(data.get("highest_education", "A Level or Equivalent")),
            "imd_band": str(data.get("imd_band", "50-60%")),
            "age_band": str(data.get("age_band", "0-35")),
            "disability": str(data.get("disability", "N")),
            "studied_credits": float(data.get("studied_credits", 60)),
            "num_of_prev_attempts": float(data.get("num_of_prev_attempts", 0)),
            "assessments_submitted": float(data.get("assessments_submitted", 4)),
            "registration_lead_time": float(data.get("registration_lead_time", 30)),
        }

        result = predict_student_outcome(reg_res["pipeline"], clf_res["pipeline"], input_dict)
        result["status"] = "success"
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/analytics")
def analytics_page():
    module_id = request.args.get("module", "ALL")
    df = get_cached_dataset(module_id)

    cls_summary, reg_summary = run_all_models_summary(df)

    return render_template(
        "analytics.html",
        cls_summary=cls_summary,
        reg_summary=reg_summary,
    )


@app.route("/api/modules", methods=["GET"])
def api_modules():
    return jsonify(get_available_modules())


if __name__ == "__main__":
    print("Pre-warming OULAD dataset cache...")
    try:
        get_cached_dataset("ALL")
    except Exception as err:
        print(f"Notice: {err}")
    app.run(debug=True, port=5000)
