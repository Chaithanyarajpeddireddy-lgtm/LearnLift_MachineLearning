from flask import Flask, render_template

from main import (
    Lasso,
    Ridge,
    load_data,
    get_dataset_summary,
    preprocess_data,
    generate_plots,
    generate_eda,
    train_linear_regression,
    train_logistic_regression,
    train_regularized_regression,
    train_decision_tree_models,
    generate_model_plots,
    optimize_data,
)

app = Flask(__name__, static_folder="Static", template_folder="templates")

DATA_PATH = "Datasets/archive/spanish_houses.csv"


@app.route("/")
@app.route("/eda")
@app.route("/models")
@app.route("/linear-regression")
@app.route("/logistic-regression")
@app.route("/preprocessing")
@app.route("/visualization")
@app.route("/optimization")
def index():
    df = load_data(DATA_PATH)
    summary = get_dataset_summary(df)
    processed, preprocess_notes = preprocess_data(df)
    plots = generate_plots(processed)
    lr_report = train_linear_regression(processed)
    logistic_report = train_logistic_regression(df)
    regularized_reports = {
        "ridge": train_regularized_regression(processed, "Ridge regression", Ridge(alpha=1.0)),
        "lasso": train_regularized_regression(processed, "Lasso regression", Lasso(alpha=0.1, max_iter=10000)),
    }
    tree_reports = train_decision_tree_models(df)
    model_plots = generate_model_plots(df, lr_report, logistic_report)
    eda = generate_eda(df)
    optimization = optimize_data(processed)

    return render_template(
        "index.html",
        summary=summary,
        preprocess_notes=preprocess_notes,
        lr_report=lr_report,
        logistic_report=logistic_report,
        regularized_reports=regularized_reports,
        tree_reports=tree_reports,
        eda=eda,
        optimization=optimization,
        plot_hist=plots.get("histogram"),
        plot_scatter=plots.get("scatter"),
        plot_heatmap=plots.get("heatmap"),
        plot_linear=model_plots.get("linear"),
        plot_logistic=model_plots.get("logistic"),
    )


if __name__ == "__main__":
    app.run(debug=True)
