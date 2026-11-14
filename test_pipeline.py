"""
=============================================================================
STAGE 7: Unified End-to-End Pipeline & System Validation Runner
=============================================================================
Script: test_pipeline.py
Project: Student Performance Predictor

Verifies all 10 project milestones:
 1. Raw and processed datasets exist, load cleanly, and have correct dimensions.
 2. Preprocessor pipeline (StandardScaler) loads and transforms features.
 3. Trained Scikit-learn Linear Regression model loads successfully.
 4. Inference works and outputs scores bounded strictly within [0, 20].
 5. Recommendation engine generates correct tiers, momentum, and study plans.
 6. Validation engine rejects missing, out-of-range, and invalid inputs.
 7. SQLite database saves successful predictions (tested with isolated DB).
 8. Cohort dashboard statistics (Total, Avg, Pass Rate, At-Risk) compute dynamically.
 9. Flask web application routes (/, /predict, /dashboard) return HTTP 200.
10. All visual analytics charts load from static/images with HTTP 200.

Rule Enforced:
Real student prediction records in data/predictions.db are fully preserved.
=============================================================================
"""

import os
import sys
import shutil
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is on the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.recommendation_engine import (
    validate_student_inputs,
    determine_performance_tier,
    analyze_grade_momentum,
    generate_recommendations
)
from src.predictor import StudentPerformancePredictor
from src.database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_dashboard_stats,
    clear_all_predictions
)
from app import app


def log_step(step_num: int, title: str):
    print(f"\n[Test {step_num}/10] {title}")


def test_1_datasets():
    log_step(1, "Verifying Dataset Files and Dimensions...")
    clean_csv = os.path.join(BASE_DIR, "data", "processed", "student_performance_clean.csv")
    train_csv = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
    test_csv = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")

    for path in [clean_csv, train_csv, test_csv]:
        assert os.path.exists(path), f"Missing dataset file: {path}"

    df_clean = pd.read_csv(clean_csv)
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    assert df_clean.shape == (395, 6), f"Unexpected clean shape: {df_clean.shape}"
    assert df_train.shape == (316, 6), f"Unexpected train shape: {df_train.shape}"
    assert df_test.shape == (79, 6), f"Unexpected test shape: {df_test.shape}"
    assert df_clean.isnull().sum().sum() == 0, "Clean dataset contains NaN values"

    print(f"         Total records: {len(df_clean)}, Train: {len(df_train)} (80%), Test: {len(df_test)} (20%).")
    print("         [PASS] Datasets verified.")


def test_2_preprocessor():
    log_step(2, "Verifying Preprocessing Pipeline...")
    scaler_path = os.path.join(BASE_DIR, "models", "preprocessor_pipeline.pkl")
    assert os.path.exists(scaler_path), f"Missing scaler pipeline: {scaler_path}"

    pipeline = joblib.load(scaler_path)
    sample_features = pd.DataFrame([{
        "studytime": 2, "failures": 0, "absences": 4, "G1": 12.0, "G2": 13.0
    }])
    transformed = pipeline.transform(sample_features)
    assert transformed.shape == (1, 5), f"Unexpected transformed shape: {transformed.shape}"
    print("         [PASS] Preprocessor pipeline loaded and transformed features correctly.")


def test_3_model():
    log_step(3, "Verifying Trained Linear Regression Model...")
    model_path = os.path.join(BASE_DIR, "models", "student_model.pkl")
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")

    assert os.path.exists(model_path), f"Missing model: {model_path}"
    assert os.path.exists(metrics_path), f"Missing metrics: {metrics_path}"

    model = joblib.load(model_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    r2 = metrics["evaluation_metrics"]["r2_score"]
    mae = metrics["evaluation_metrics"]["mae"]
    assert r2 > 0.75, f"R2 score unexpectedly low: {r2}"
    assert mae < 2.0, f"MAE unexpectedly high: {mae}"

    print(f"         Model R2: {r2:.4f} ({metrics['evaluation_metrics']['r2_percentage']}%), MAE: {mae:.4f} marks.")
    print("         [PASS] Model and metrics metadata verified.")


def test_4_prediction():
    log_step(4, "Verifying Model Inference & Bounded Scores...")
    predictor = StudentPerformancePredictor()
    student = {"studytime": 3, "failures": 0, "absences": 2, "G1": 15.0, "G2": 16.0}
    res = predictor.predict(student)

    assert "predicted_score" in res, "Missing predicted_score in result"
    score = res["predicted_score"]
    assert 0.0 <= score <= 20.0, f"Score out of range [0, 20]: {score}"
    assert isinstance(score, float), f"Expected float, got {type(score)}"

    print(f"         Sample prediction for G1=15, G2=16 -> {score}/20 (Raw: {res['raw_prediction']}).")
    print("         [PASS] Inference works and is safely bounded within [0, 20].")


def test_5_recommendations():
    log_step(5, "Verifying Recommendation Engine Logic...")
    student = {"studytime": 1, "failures": 2, "absences": 16, "G1": 6.0, "G2": 6.0}
    recs = generate_recommendations(student, predicted_score=5.5)

    assert recs["performance_tier"] == "At-Risk", f"Expected At-Risk, got {recs['performance_tier']}"
    assert recs["grade_momentum"] == "Plateaued", f"Expected Plateaued, got {recs['grade_momentum']}"
    assert len(recs["study_recommendations"]) > 0, "Study recommendations empty"
    assert len(recs["attendance_recommendations"]) > 0, "Attendance recommendations empty"
    assert len(recs["overall_improvement_plan"]) >= 3, "Roadmap plan has fewer than 3 steps"

    print(f"         Tier: {recs['performance_tier']}, Momentum: {recs['grade_momentum']}, Plan steps: {len(recs['overall_improvement_plan'])}.")
    print("         [PASS] Recommendation engine logic verified.")


def test_6_input_validation():
    log_step(6, "Verifying Input Validation & Rejection of Bad Inputs...")
    bad_inputs = [
        {"studytime": 9, "failures": 0, "absences": 2, "G1": 10, "G2": 10},   # Invalid study time
        {"studytime": 2, "failures": -1, "absences": 2, "G1": 10, "G2": 10},  # Negative failures
        {"studytime": 2, "failures": 0, "absences": -5, "G1": 10, "G2": 10},  # Negative absences
        {"studytime": 2, "failures": 0, "absences": 2, "G1": 25.0, "G2": 10}, # G1 > 20
        {"studytime": 2, "failures": 0, "absences": 2, "G1": 10, "G2": -1.0}, # G2 < 0
        {"studytime": None, "failures": 0, "absences": 2, "G1": 10, "G2": 10}  # Missing key
    ]

    for bad in bad_inputs:
        try:
            validate_student_inputs(bad)
            raise AssertionError(f"Failed to reject invalid input: {bad}")
        except ValueError:
            pass  # Expected

    print(f"         Successfully rejected {len(bad_inputs)} out-of-bounds / malformed inputs.")
    print("         [PASS] Input validation verified.")


def test_7_database_persistence():
    log_step(7, "Verifying SQLite Persistence in Isolated Test DB...")
    test_db = os.path.join(BASE_DIR, "data", "test_isolated.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    init_db(test_db)
    rec_id = save_prediction(
        studytime=2, failures=0, absences=3, g1=14.0, g2=15.0,
        predicted_score=15.2, performance_tier="Merit", grade_momentum="Improving",
        db_path=test_db
    )
    assert rec_id == 1, f"Expected inserted id 1, got {rec_id}"

    records = get_all_predictions(test_db)
    assert len(records) == 1, f"Expected 1 record, found {len(records)}"
    assert records[0]["predicted_score"] == 15.2
    assert records[0]["performance_tier"] == "Merit"

    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)

    print("         Isolated SQLite record inserted and verified with zero personal identifiers.")
    print("         [PASS] Database persistence verified.")


def test_8_dashboard_stats():
    log_step(8, "Verifying Dynamic Dashboard Statistics Calculation...")
    test_db = os.path.join(BASE_DIR, "data", "test_stats.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    init_db(test_db)

    # Insert 4 test records: 3 passing, 1 failing
    save_prediction(2, 0, 2, 16.0, 17.0, 17.5, "Distinction", "Improving", db_path=test_db)
    save_prediction(2, 0, 4, 14.0, 14.0, 14.2, "Merit", "Plateaued", db_path=test_db)
    save_prediction(2, 0, 6, 10.0, 11.0, 10.8, "Pass", "Improving", db_path=test_db)
    save_prediction(1, 2, 12, 6.0, 6.0, 5.0, "At-Risk", "Plateaued", db_path=test_db)

    stats = get_dashboard_stats(test_db)
    assert stats["total_students"] == 4, f"Expected total 4, got {stats['total_students']}"
    assert stats["at_risk_count"] == 1, f"Expected 1 at-risk, got {stats['at_risk_count']}"
    assert stats["pass_rate"] == 75.0, f"Expected 75.0% pass rate, got {stats['pass_rate']}%"
    expected_avg = round((17.5 + 14.2 + 10.8 + 5.0) / 4, 2)
    assert stats["avg_predicted_grade"] == expected_avg, f"Expected avg {expected_avg}, got {stats['avg_predicted_grade']}"

    if os.path.exists(test_db):
        os.remove(test_db)

    print(f"         Dynamic Stats: Total=4, Pass Rate=75.0%, Avg={expected_avg}/20, At-Risk=1.")
    print("         [PASS] Dynamic statistics calculations verified.")


def test_9_flask_routes():
    log_step(9, "Verifying Flask Application Routes (/, /predict, /dashboard)...")
    
    # Backup existing real predictions database to preserve user data
    real_db = os.path.join(BASE_DIR, "data", "predictions.db")
    bak_db = os.path.join(BASE_DIR, "data", "predictions.db.bak")
    has_real_db = os.path.exists(real_db)
    if has_real_db:
        shutil.copyfile(real_db, bak_db)

    try:
        client = app.test_client()

        # Route 1: GET /
        resp_home = client.get("/")
        assert resp_home.status_code == 200, f"GET / returned {resp_home.status_code}"
        assert "Student Academic Information" in resp_home.get_data(as_text=True)

        # Route 2: POST /predict
        resp_pred = client.post("/predict", data={
            "studytime": "2", "failures": "0", "absences": "2", "G1": "14.0", "G2": "15.0"
        })
        assert resp_pred.status_code == 200, f"POST /predict returned {resp_pred.status_code}"
        pred_html = resp_pred.get_data(as_text=True)
        assert "Predicted Final Exam Score" in pred_html
        assert "Personalized Action &amp; Advisory Plan" in pred_html or "Personalized Action & Advisory Plan" in pred_html

        # Route 3: GET /dashboard
        resp_dash = client.get("/dashboard")
        assert resp_dash.status_code == 200, f"GET /dashboard returned {resp_dash.status_code}"
        dash_html = resp_dash.get_data(as_text=True)
        assert "Teacher Dashboard &amp; Cohort Overview" in dash_html or "Teacher Dashboard & Cohort Overview" in dash_html

        print("         All routes (/, /predict, /dashboard) returned HTTP 200 with complete rendered HTML.")
        print("         [PASS] Flask web routing verified.")

    finally:
        # Restore real predictions database so user records are preserved
        if has_real_db and os.path.exists(bak_db):
            shutil.copyfile(bak_db, real_db)
            os.remove(bak_db)
            print("         [Preserved] Original predictions database restored intact.")


def test_10_charts():
    log_step(10, "Verifying Dashboard Static Diagnostic Charts...")
    charts = [
        "final_grade_distribution.png",
        "correlation_heatmap.png",
        "study_time_vs_g3.png",
        "absences_vs_g3.png",
        "actual_vs_predicted.png"
    ]
    client = app.test_client()

    for chart in charts:
        chart_path = os.path.join(BASE_DIR, "static", "images", chart)
        assert os.path.exists(chart_path), f"Chart file missing on disk: {chart_path}"

        url = f"/static/images/{chart}"
        resp = client.get(url)
        assert resp.status_code == 200, f"Failed HTTP 200 on {url}"
        assert len(resp.data) > 10000, f"Chart payload suspiciously small for {chart}"
        print(f"         Loaded: {chart:30s} (HTTP 200, {len(resp.data):,} bytes)")

    print("         [PASS] All 5 diagnostic charts verified.")


def run_all_tests():
    print("=" * 75)
    print(" STUDENT PERFORMANCE PREDICTOR — END-TO-END PIPELINE VALIDATION")
    print("=" * 75)

    test_1_datasets()
    test_2_preprocessor()
    test_3_model()
    test_4_prediction()
    test_5_recommendations()
    test_6_input_validation()
    test_7_database_persistence()
    test_8_dashboard_stats()
    test_9_flask_routes()
    test_10_charts()

    print("\n" + "=" * 75)
    print(" ALL 10 PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print(" SYSTEM IS PRODUCTION-READY AND FULLY FUNCTIONAL.")
    print("=" * 75)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as err:
        print(f"\n[FATAL ERROR] Pipeline validation failed: {err}", file=sys.stderr)
        sys.exit(1)
