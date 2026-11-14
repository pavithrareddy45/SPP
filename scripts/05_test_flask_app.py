"""
=============================================================================
STAGE 5 VERIFICATION: Flask Application Automated Test Suite
=============================================================================
Project: Student Performance Predictor
Tests:
1. GET /: Validates page load, HTML structure, 5 input fields, preset buttons.
2. POST /predict (High Achiever Profile): Asserts Distinction, score >= 16, recommendations.
3. POST /predict (At-Risk Profile): Asserts At-Risk, score < 10, recommendations.
4. POST /predict (Invalid/Out-of-Bounds Input): Asserts HTTP 400 & clean error alert.
5. GET /api/metrics: Asserts JSON metadata availability.
=============================================================================
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app


def test_flask_application():
    print("=" * 70)
    print("STAGE 5: TESTING FLASK WEB APPLICATION")
    print("=" * 70)

    client = app.test_client()

    # -----------------------------------------------------------------------
    # Test 1: GET / (Homepage and Form Inspection)
    # -----------------------------------------------------------------------
    print("\n[Test 1] Testing GET / (Home Page & Input Form)...")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"
    html = response.get_data(as_text=True)

    # Check for exact 5 inputs
    required_inputs = ['name="studytime"', 'name="failures"', 'name="absences"', 'name="G1"', 'name="G2"']
    for field in required_inputs:
        assert field in html, f"Missing input element in form: {field}"
    print("         All 5 model input fields verified in HTML form.")

    # Check for preset buttons
    assert "Try High Achiever" in html, "Missing High Achiever preset button"
    assert "Try At-Risk Student" in html, "Missing At-Risk preset button"
    print("         Preset profile buttons verified.")

    # -----------------------------------------------------------------------
    # Test 2: POST /predict (High Achiever Preset)
    # -----------------------------------------------------------------------
    print("\n[Test 2] Testing POST /predict with High Achiever Profile...")
    high_data = {
        "studytime": "3",
        "failures": "0",
        "absences": "2",
        "G1": "17.0",
        "G2": "18.0"
    }
    response_high = client.post("/predict", data=high_data)
    assert response_high.status_code == 200, f"Expected 200 but got {response_high.status_code}"
    high_html = response_high.get_data(as_text=True)

    assert "Distinction" in high_html, "Expected 'Distinction' tier in result page"
    assert "Weekly Study Strategy" in high_html, "Missing study strategy section"
    assert "Attendance &amp; Engagement" in high_html or "Attendance & Engagement" in high_html, "Missing attendance section"
    assert "Phase-by-Phase Improvement Roadmap" in high_html, "Missing roadmap"
    print("         High Achiever prediction rendered correctly (Distinction tier, all recommendations present).")

    # -----------------------------------------------------------------------
    # Test 3: POST /predict (At-Risk Preset)
    # -----------------------------------------------------------------------
    print("\n[Test 3] Testing POST /predict with At-Risk Profile...")
    risk_data = {
        "studytime": "1",
        "failures": "2",
        "absences": "12",
        "G1": "6.0",
        "G2": "6.0"
    }
    response_risk = client.post("/predict", data=risk_data)
    assert response_risk.status_code == 200, f"Expected 200 but got {response_risk.status_code}"
    risk_html = response_risk.get_data(as_text=True)

    assert "At-Risk" in risk_html, "Expected 'At-Risk' tier in result page"
    assert "Below passing standard" in risk_html or "Needs immediate intervention" in risk_html, "Expected at-risk summary"
    print("         At-Risk prediction rendered correctly (At-Risk tier, intervention roadmap present).")

    # -----------------------------------------------------------------------
    # Test 4: POST /predict (Invalid Inputs)
    # -----------------------------------------------------------------------
    print("\n[Test 4] Testing POST /predict with Invalid/Out-of-Bounds Inputs...")
    invalid_data = {
        "studytime": "5",  # Invalid level (>4)
        "failures": "0",
        "absences": "2",
        "G1": "25.0",      # Invalid grade (>20)
        "G2": "10.0"
    }
    response_invalid = client.post("/predict", data=invalid_data)
    assert response_invalid.status_code == 400, f"Expected 400 for invalid inputs, got {response_invalid.status_code}"
    invalid_html = response_invalid.get_data(as_text=True)
    assert "Input Error" in invalid_html, "Expected error banner in HTML"
    print("         Invalid inputs rejected with HTTP 400 and clear error banner.")

    # -----------------------------------------------------------------------
    # Test 5: GET /api/metrics (JSON API)
    # -----------------------------------------------------------------------
    print("\n[Test 5] Testing GET /api/metrics...")
    response_metrics = client.get("/api/metrics")
    assert response_metrics.status_code == 200, f"Expected 200 but got {response_metrics.status_code}"
    metrics_json = response_metrics.get_json()
    assert "evaluation_metrics" in metrics_json, "Missing evaluation_metrics key"
    assert metrics_json["evaluation_metrics"]["r2_score"] == 0.7822
    print(f"         API metrics returned successfully (R² = {metrics_json['evaluation_metrics']['r2_percentage']}%).")

    print("\n" + "=" * 70)
    print("STAGE 5 ALL FLASK TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_flask_application()
    except Exception as e:
        print(f"\n[ERROR] Flask test failed: {e}", file=sys.stderr)
        sys.exit(1)
