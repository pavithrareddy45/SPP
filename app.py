"""
=============================================================================
STAGE 6: Flask Web Application with Teacher Analytics Dashboard
=============================================================================
Application entry point: app.py
Project: Student Performance Predictor

Features:
- Student exam score prediction and personalized advisory.
- Teacher Analytics Dashboard (/dashboard) with dynamic SQLite statistics:
  1. Total Students Analyzed
  2. Average Predicted Final Grade
  3. Class Pass Rate (predicted >= 10)
  4. At-Risk Students (predicted < 10)
- Visual analytics embedding verified EDA and Regression diagnostic charts.
- Filterable student summary table (Distinction, Merit, Pass, At-Risk).
- Pinterest-inspired pastel aesthetic throughout.
=============================================================================
"""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from src.predictor import StudentPerformancePredictor
from src.recommendation_engine import validate_student_inputs
from src.database import init_db, save_prediction, get_all_predictions, get_dashboard_stats

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "student_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "preprocessor_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Initialize SQLite database
init_db()

# Initialize the predictor (loads student_model.pkl and preprocessor_pipeline.pkl)
try:
    predictor = StudentPerformancePredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    print(" [OK] Model and preprocessor loaded successfully.")
except Exception as e:
    print(f" [ERROR] Failed to load model or preprocessor: {e}")
    predictor = None

# Load model evaluation metrics for display
model_metrics = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)
    except Exception as e:
        print(f" [WARNING] Could not load model_metrics.json: {e}")


@app.route("/", methods=["GET"])
def index():
    """
    Renders the student input form with sample profile triggers and model information.
    """
    return render_template(
        "index.html",
        metrics=model_metrics.get("evaluation_metrics", {}),
        form_data={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission, runs validation, executes inference and recommendation
    generation, saves the record to SQLite, and renders the result view.
    """
    if predictor is None:
        flash("Error: The machine learning model is not loaded. Please ensure models are trained.", "danger")
        return redirect(url_for("index"))

    form_data = request.form.to_dict()

    try:
        # 1. Parse and validate inputs
        student_data = {
            "studytime": form_data.get("studytime"),
            "failures": form_data.get("failures"),
            "absences": form_data.get("absences"),
            "G1": form_data.get("G1"),
            "G2": form_data.get("G2"),
        }

        # Validate with the decoupled validation engine
        validated = validate_student_inputs(student_data)

        # 2. Run model inference & recommendation generation
        result = predictor.predict(validated)

        # 3. Save successful prediction record to SQLite (no personal identifiers)
        save_prediction(
            studytime=validated["studytime"],
            failures=validated["failures"],
            absences=validated["absences"],
            g1=validated["G1"],
            g2=validated["G2"],
            predicted_score=result["predicted_score"],
            performance_tier=result["performance_tier"],
            grade_momentum=result["grade_momentum"]
        )

        # 4. Render beautiful results page
        return render_template(
            "result.html",
            result=result,
            student=validated
        )

    except ValueError as val_err:
        flash(f"Validation Error: {str(val_err)}", "error")
        return render_template("index.html", form_data=form_data, error=str(val_err)), 400

    except Exception as exc:
        flash(f"An unexpected error occurred during prediction: {str(exc)}", "error")
        return render_template("index.html", form_data=form_data, error=str(exc)), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Renders the Teacher Analytics Dashboard with dynamic cohort metrics,
    verified visual analytics charts, and filterable student prediction history.
    """
    stats = get_dashboard_stats()
    records = get_all_predictions()

    return render_template(
        "dashboard.html",
        stats=stats,
        records=records,
        metrics=model_metrics.get("evaluation_metrics", {})
    )


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns the model evaluation metrics as JSON."""
    return jsonify(model_metrics)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns dynamic dashboard statistics as JSON."""
    return jsonify(get_dashboard_stats())


@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", error="The requested page was not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", error="Internal server error occurred."), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
