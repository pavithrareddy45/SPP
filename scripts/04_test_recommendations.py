"""
=============================================================================
STAGE 4 VERIFICATION: Test Suite for Recommendation Engine & Predictor
=============================================================================
Project: Student Performance Predictor
Tests:
1. At-Risk Student Profile
2. Declining Student Profile
3. Consistent High-Achiever Profile
4. Improving Student Profile
5. Student with High Absences Profile
6. Input Validation & Error Handling (Missing fields, negative values, out-of-range)
=============================================================================
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.recommendation_engine import (
    validate_student_inputs,
    determine_performance_tier,
    analyze_grade_momentum,
    generate_recommendations
)
from src.predictor import StudentPerformancePredictor


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


def test_student_profiles():
    """Tests the 5 required student profiles through the unified predictor & recommendation engine."""
    print_banner("1. Testing Required Student Profiles")
    predictor = StudentPerformancePredictor()

    test_cases = [
        {
            "name": "At-Risk Student",
            "data": {"studytime": 1, "failures": 2, "absences": 12, "G1": 6.0, "G2": 6.0},
            "expected_tier": "At-Risk",
            "expected_momentum": "Plateaued"
        },
        {
            "name": "Declining Student",
            "data": {"studytime": 2, "failures": 0, "absences": 6, "G1": 15.0, "G2": 10.0},
            "expected_tier": "Pass",
            "expected_momentum": "Declining"
        },
        {
            "name": "Consistent High-Achiever",
            "data": {"studytime": 3, "failures": 0, "absences": 2, "G1": 17.0, "G2": 18.0},
            "expected_tier": "Distinction",
            "expected_momentum": "Plateaued"
        },
        {
            "name": "Improving Student",
            "data": {"studytime": 2, "failures": 0, "absences": 4, "G1": 8.0, "G2": 13.0},
            "expected_tier": "Pass",
            "expected_momentum": "Improving"
        },
        {
            "name": "Student with High Absences",
            "data": {"studytime": 2, "failures": 0, "absences": 24, "G1": 12.0, "G2": 12.0},
            "expected_tier": "Pass",
            "expected_momentum": "Plateaued"
        }
    ]

    for case in test_cases:
        name = case["name"]
        data = case["data"]
        result = predictor.predict(data)

        print(f"\n>> Profile: [{name}]")
        print(f"   Inputs: StudyTime={data['studytime']}, Failures={data['failures']}, Absences={data['absences']}, G1={data['G1']}, G2={data['G2']}")
        print(f"   Predicted Final Exam Score (G3): {result['predicted_score']} / 20 (Raw: {result['raw_prediction']})")
        print(f"   Performance Tier:               {result['performance_tier']} - {result['tier_details']['summary']}")
        print(f"   Grade Momentum:                 {result['grade_momentum']} ({result['momentum_details']['delta']:+.1f} marks)")
        print(f"   Study Advice:                   {result['study_recommendations'][0]}")
        print(f"   Attendance Advice:              {result['attendance_recommendations'][0]}")
        print("   Action Roadmap:")
        for idx, step in enumerate(result['overall_improvement_plan'], 1):
            print(f"     {idx}. {step}")

        # Assertions
        assert result["performance_tier"] == case["expected_tier"], (
            f"Expected tier {case['expected_tier']} but got {result['performance_tier']}"
        )
        assert result["grade_momentum"] == case["expected_momentum"], (
            f"Expected momentum {case['expected_momentum']} but got {result['grade_momentum']}"
        )
        print("   [PASS] Profile validation asserted successfully.")


def test_input_validation():
    """Tests handling of invalid, missing, and out-of-range inputs."""
    print_banner("2. Testing Input Validation & Error Handling")

    invalid_cases = [
        ({"studytime": None, "failures": 0, "absences": 2, "G1": 10, "G2": 12}, "Missing required input field: 'studytime'"),
        ({"studytime": 5, "failures": 0, "absences": 2, "G1": 10, "G2": 12}, "Invalid studytime: 5"),
        ({"studytime": 2, "failures": -1, "absences": 2, "G1": 10, "G2": 12}, "Invalid failures: -1"),
        ({"studytime": 2, "failures": 0, "absences": -5, "G1": 10, "G2": 12}, "Invalid absences: -5"),
        ({"studytime": 2, "failures": 0, "absences": 4, "G1": 25, "G2": 12}, "Invalid G1 score: 25.0"),
        ({"studytime": 2, "failures": 0, "absences": 4, "G1": 10, "G2": -2}, "Invalid G2 score: -2.0"),
        ({"studytime": "invalid", "failures": 0, "absences": 4, "G1": 10, "G2": 12}, "All student inputs must be valid numbers")
    ]

    for data, expected_error in invalid_cases:
        try:
            validate_student_inputs(data)
            raise AssertionError(f"Expected failure for input: {data}")
        except ValueError as err:
            assert expected_error in str(err), f"Expected '{expected_error}' in '{str(err)}'"
            print(f"   [PASS] Successfully caught invalid input: {err}")


if __name__ == "__main__":
    try:
        test_student_profiles()
        test_input_validation()
        print("\n" + "=" * 70)
        print("STAGE 4 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Stage 4 test failed: {e}", file=sys.stderr)
        sys.exit(1)
