"""
=============================================================================
STAGE 6 VERIFICATION: Teacher Dashboard & Visual Analytics Test Suite
=============================================================================
Project: Student Performance Predictor
Tests:
1. Dashboard loads successfully on GET /dashboard.
2. Empty database displays correctly with zero-state guidance.
3. Successful predictions are saved to SQLite (no personal identifiers).
4. Cohort statistics are calculated dynamically (Total, Average, Pass Rate, At-Risk).
5. Student summary table contains correct tier data attributes for client-side filtering.
6. All 5 verified visual analytics diagnostic charts load with HTTP 200.
7. Existing prediction functionality and error handling remain uncorrupted.
=============================================================================
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app
from src.database import clear_all_predictions, get_all_predictions, get_dashboard_stats


def test_dashboard_system():
    print("=" * 70)
    print("STAGE 6: TESTING TEACHER DASHBOARD & DYNAMIC ANALYTICS")
    print("=" * 70)

    client = app.test_client()

    # -----------------------------------------------------------------------
    # Test 1: Empty Database State
    # -----------------------------------------------------------------------
    print("\n[Test 1] Testing Empty Database State on GET /dashboard...")
    clear_all_predictions()
    empty_stats = get_dashboard_stats()
    assert empty_stats["total_students"] == 0, "Expected 0 students in empty db"
    assert empty_stats["has_data"] is False, "Expected has_data=False"

    response_empty = client.get("/dashboard")
    assert response_empty.status_code == 200, f"Expected 200, got {response_empty.status_code}"
    empty_html = response_empty.get_data(as_text=True)
    assert "No Student Evaluations Recorded Yet" in empty_html, "Missing empty state banner"
    assert "N/A" in empty_html or "0" in empty_html, "Expected N/A or 0 for empty averages"
    print("         Empty database renders gracefully with zero-state banner.")

    # -----------------------------------------------------------------------
    # Test 2: Saving Successful Predictions to SQLite
    # -----------------------------------------------------------------------
    print("\n[Test 2] Testing Persistence of Successful Predictions...")
    
    # 1. High Achiever (Distinction, score ~ 18.3)
    p1 = client.post("/predict", data={
        "studytime": "3", "failures": "0", "absences": "2", "G1": "17.0", "G2": "18.0"
    })
    assert p1.status_code == 200

    # 2. At-Risk Student (At-Risk, score ~ 4.6)
    p2 = client.post("/predict", data={
        "studytime": "1", "failures": "2", "absences": "12", "G1": "6.0", "G2": "6.0"
    })
    assert p2.status_code == 200

    # 3. Passing Student (Pass, score ~ 12.0)
    p3 = client.post("/predict", data={
        "studytime": "2", "failures": "0", "absences": "4", "G1": "11.0", "G2": "12.0"
    })
    assert p3.status_code == 200

    # Verify records in database
    records = get_all_predictions()
    assert len(records) == 3, f"Expected 3 records in DB, found {len(records)}"
    print(f"         Successfully saved {len(records)} prediction records in SQLite.")

    # -----------------------------------------------------------------------
    # Test 3: Dynamic Statistics Calculation
    # -----------------------------------------------------------------------
    print("\n[Test 3] Testing Dynamic Cohort Statistics Calculation...")
    stats = get_dashboard_stats()
    assert stats["total_students"] == 3, f"Expected total 3, got {stats['total_students']}"
    assert stats["at_risk_count"] == 1, f"Expected 1 at-risk student, got {stats['at_risk_count']}"
    assert stats["distinction_count"] == 1, f"Expected 1 distinction, got {stats['distinction_count']}"
    # Pass count: 2 students have score >= 10.0, so pass rate is 2/3 = 66.7%
    expected_pass_rate = round((2 / 3) * 100, 1)
    assert stats["pass_rate"] == expected_pass_rate, f"Expected pass rate {expected_pass_rate}%, got {stats['pass_rate']}%"
    print(f"         Dynamic Stats: Total={stats['total_students']}, Avg={stats['avg_predicted_grade']}/20, PassRate={stats['pass_rate']}%, AtRisk={stats['at_risk_count']}.")

    # -----------------------------------------------------------------------
    # Test 4: Dashboard UI & Tier Filters
    # -----------------------------------------------------------------------
    print("\n[Test 4] Testing Dashboard UI & Filter Attributes...")
    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == 200
    dash_html = dash_resp.get_data(as_text=True)

    # Check summary cards in HTML
    assert f"{stats['total_students']}" in dash_html
    assert f"{stats['pass_rate']}%" in dash_html
    assert f"{stats['at_risk_count']}" in dash_html

    # Check table rows have data-tier attributes for client-side filter
    assert 'data-tier="Distinction"' in dash_html, "Missing data-tier='Distinction' row"
    assert 'data-tier="At-Risk"' in dash_html, "Missing data-tier='At-Risk' row"
    assert 'data-tier="Pass"' in dash_html, "Missing data-tier='Pass' row"
    assert 'id="tierFilterGroup"' in dash_html, "Missing filter buttons group"
    print("         Dashboard HTML table populated with proper tier filtering attributes.")

    # -----------------------------------------------------------------------
    # Test 5: Visual Analytics Charts Loading
    # -----------------------------------------------------------------------
    print("\n[Test 5] Testing Static Visual Analytics Charts...")
    charts = [
        "/static/images/final_grade_distribution.png",
        "/static/images/correlation_heatmap.png",
        "/static/images/study_time_vs_g3.png",
        "/static/images/absences_vs_g3.png",
        "/static/images/actual_vs_predicted.png"
    ]
    for chart_url in charts:
        img_resp = client.get(chart_url)
        assert img_resp.status_code == 200, f"Failed to load chart: {chart_url} (status: {img_resp.status_code})"
        assert len(img_resp.data) > 10000, f"Chart data seems too small or corrupt: {chart_url}"
        print(f"         Chart verified: {chart_url} (HTTP 200, {len(img_resp.data):,} bytes)")

    # -----------------------------------------------------------------------
    # Test 6: Preserved Functionality & Input Error Handling
    # -----------------------------------------------------------------------
    print("\n[Test 6] Testing Preservation of Predictor & Error Handling...")
    invalid_resp = client.post("/predict", data={
        "studytime": "2", "failures": "0", "absences": "-10", "G1": "10", "G2": "10"
    })
    assert invalid_resp.status_code == 400, "Expected 400 for negative absences"
    # Ensure invalid prediction was NOT saved in database
    records_after_invalid = get_all_predictions()
    assert len(records_after_invalid) == 3, "Invalid prediction should NOT be saved in DB"
    print("         Invalid prediction successfully rejected and excluded from database.")

    print("\n" + "=" * 70)
    print("STAGE 6 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_dashboard_system()
    except Exception as e:
        print(f"\n[ERROR] Stage 6 test failed: {e}", file=sys.stderr)
        sys.exit(1)
