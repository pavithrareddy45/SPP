"""
=============================================================================
Inference Predictor Interface
=============================================================================
Module: src/predictor.py
Project: Student Performance Predictor

Provides a clean, beginner-friendly inference wrapper that loads the serialized
Scikit-learn model and preprocessor, predicts the final examination score (G3),
and coordinates with the recommendation engine.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.recommendation_engine import generate_recommendations, validate_student_inputs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "preprocessor_pipeline.pkl")


class StudentPerformancePredictor:
    """
    Inference class to handle model loading, feature scaling, prediction,
    and personalized recommendation generation.
    """

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """Loads the pre-trained model and preprocessing pipeline into memory."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler pipeline not found at: {scaler_path}.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = ["studytime", "failures", "absences", "G1", "G2"]

    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs full inference and advisory pipeline for a single student.

        Steps:
        1. Validates input values and educational bounds.
        2. Formats features into a DataFrame matching training schema.
        3. Standardizes features with the pre-fitted scaler.
        4. Calculates predicted G3 score and bounds it to [0, 20].
        5. Invokes recommendation engine to generate tailored action items.

        Returns:
            dict: Unified inference result containing predictions and recommendations.
        """
        # Validate inputs
        validated = validate_student_inputs(student_data)

        # Convert to 2D DataFrame with correct column order
        input_df = pd.DataFrame([{
            "studytime": validated["studytime"],
            "failures": validated["failures"],
            "absences": validated["absences"],
            "G1": validated["G1"],
            "G2": validated["G2"]
        }], columns=self.feature_columns)

        # Transform using pre-fitted scaler
        scaled_features = self.scaler.transform(input_df)

        # Raw continuous prediction
        raw_pred = float(self.model.predict(scaled_features)[0])

        # Academically bounded prediction [0, 20]
        bounded_pred = max(0.0, min(20.0, round(raw_pred, 2)))

        # Generate rule-based recommendations
        recs = generate_recommendations(validated, bounded_pred)

        return {
            "inputs": validated,
            "raw_prediction": round(raw_pred, 3),
            "predicted_score": bounded_pred,
            "performance_tier": recs["performance_tier"],
            "tier_details": recs["tier_details"],
            "grade_momentum": recs["grade_momentum"],
            "momentum_details": recs["momentum_details"],
            "study_recommendations": recs["study_recommendations"],
            "attendance_recommendations": recs["attendance_recommendations"],
            "overall_improvement_plan": recs["overall_improvement_plan"]
        }
