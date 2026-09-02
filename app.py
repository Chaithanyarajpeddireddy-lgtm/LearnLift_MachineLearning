import os
from flask import Flask, jsonify, render_template, request

from main import (
    generate_eda,
    generate_plots,
    get_available_regions,
    get_dataset_summary,
    load_data,
    optimize_data,
    predict_house_price,
    train_classification_models,
    train_regression_models,
)

app = Flask(__name__, static_folder="Static", template_folder="templates")

# In-memory store for loaded data and trained pipelines per region
MODEL_REGISTRY = {}


def get_or_train_models(region_file):
    """Retrieve cached models and dataset stats or train them once."""
    if region_file in MODEL_REGISTRY:
        return MODEL_REGISTRY[region_file]

    try:
        df = load_data(region_file)
    except Exception:
        df = load_data("spanish_houses.csv")
        region_file = "spanish_houses.csv"

    summary = get_dataset_summary(df)
    eda = generate_eda(df)
    plots = generate_plots(df)
    reg_results, reg_pipelines = train_regression_models(df)
    clf_results, clf_pipelines = train_classification_models(df)
    optimization = optimize_data(df)

    # Cache everything in memory for zero-latency queries
    context = {
        "region_file": region_file,
        "summary": summary,
        "eda": eda,
        "plots": plots,
        "reg_results": reg_results,
        "clf_results": clf_results,
        "optimization": optimization,
        "reg_pipelines": reg_pipelines,
        "clf_pipelines": clf_pipelines,
    }

    MODEL_REGISTRY[region_file] = context
    return context


@app.route("/")
@app.route("/dashboard")
def index():
    selected_region = request.args.get("region", "spanish_houses.csv")
    available_regions = get_available_regions()
    
    # Ensure selected_region is valid
    valid_ids = [r["id"] for r in available_regions]
    if selected_region not in valid_ids:
        selected_region = "spanish_houses.csv"

    context = get_or_train_models(selected_region)

    return render_template(
        "index.html",
        available_regions=available_regions,
        selected_region=selected_region,
        summary=context["summary"],
        eda=context["eda"],
        plots=context["plots"],
        reg_results=context["reg_results"],
        clf_results=context["clf_results"],
        optimization=context["optimization"],
    )


@app.route("/api/regions", methods=["GET"])
def api_regions():
    """Return JSON list of available regional datasets."""
    return jsonify(get_available_regions())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Real-time inference API for predicting house price and market tier.
    Accepts JSON payload with property specifications.
    """
    try:
        data = request.get_json(force=True) or request.form.to_dict()
        region_file = data.get("region", "spanish_houses.csv")
        
        context = get_or_train_models(region_file)
        reg_pipeline = context["reg_pipelines"].get("ridge") or context["reg_pipelines"].get("linear")
        clf_pipeline = context["clf_pipelines"].get("logistic") or context["clf_pipelines"].get("dt_gini")

        if not reg_pipeline or not clf_pipeline:
            return jsonify({"status": "error", "message": "Models are not yet ready."}), 500

        # Construct input vector with proper types
        input_dict = {
            "m2_real": float(data.get("m2_real", 100)),
            "room_num": float(data.get("room_num", 3)),
            "bath_num": float(data.get("bath_num", 2)),
            "swimming_pool": int(data.get("swimming_pool", 0)),
            "garden": int(data.get("garden", 0)),
            "terrace": int(data.get("terrace", 0)),
            "lift": int(data.get("lift", 0)),
            "air_conditioner": int(data.get("air_conditioner", 0)),
            "house_type": str(data.get("house_type", "Piso")),
            "condition": str(data.get("condition", "segunda mano/buen estado")),
        }

        result = predict_house_price(reg_pipeline, clf_pipeline, input_dict)
        result["status"] = "success"
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    # Pre-warm default cache on start
    print("Pre-warming model cache for national dataset...")
    try:
        get_or_train_models("spanish_houses.csv")
    except Exception as err:
        print(f"Notice: {err}")
    app.run(debug=True, port=5000)
