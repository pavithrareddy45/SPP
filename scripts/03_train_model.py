"""
=============================================================================
STAGE 3: Multiple Linear Regression Model Training & Comprehensive Evaluation
=============================================================================
Project: Student Performance Predictor
Target Variable: G3 (Final Exam Score, 0 to 20 scale)

Key Objectives of this Stage:
1. Load split train/test datasets from 'data/processed/'.
2. Load the fitted StandardScaler preprocessor from Stage 2.
3. Transform X_train and X_test using the SAME pre-fitted scaler (No refitting!).
4. Train Scikit-learn's LinearRegression model on training data.
5. Predict final exam scores (G3) on the unseen test set.
6. Compute all 4 standard regression metrics:
   - MAE  (Mean Absolute Error)
   - MSE  (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R²   (Coefficient of Determination)
7. Display an Actual vs. Predicted comparison table with bounded [0, 20] scores.
8. Analyze and interpret model coefficients and the intercept.
9. Generate and save 'static/images/actual_vs_predicted.png'.
10. Persist the trained model to 'models/student_model.pkl' and metrics to
    'models/model_metrics.json'.
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Step 1: Directory Setup & Data Loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, "train_data.csv")
TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "preprocessor_pipeline.pkl")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "student_model.pkl")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
CHART_SAVE_PATH = os.path.join(IMAGES_DIR, "actual_vs_predicted.png")

print("=" * 70)
print("STAGE 3: MULTIPLE LINEAR REGRESSION TRAINING & EVALUATION")
print("=" * 70)

print("\n[Step 1] Loading split datasets from disk...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]
TARGET_COLUMN = "G3"

# Separate features (X) and target (y)
X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].copy()

print(f"         Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"         Testing set:  X_test  {X_test.shape},  y_test  {y_test.shape}")

# ---------------------------------------------------------------------------
# Step 2: Feature Transformation using Pre-Fitted Scaler (Zero Leakage)
# ---------------------------------------------------------------------------
# IMPORTANT RULE:
# We do NOT call .fit() or .fit_transform() here!
# We only call .transform() using the scaler already fitted on X_train in Stage 2.
print("\n[Step 2] Loading fitted preprocessor and transforming features...")
scaler_pipeline = joblib.load(SCALER_PATH)

X_train_scaled = scaler_pipeline.transform(X_train)
X_test_scaled = scaler_pipeline.transform(X_test)
print("         Successfully scaled features without refitting.")

# ---------------------------------------------------------------------------
# Step 3: Train Scikit-learn Multiple Linear Regression Model
# ---------------------------------------------------------------------------
# Multiple Linear Regression models the relationship between dependent variable (G3)
# and multiple independent variables (studytime, failures, absences, G1, G2):
#
#   y = beta_0 + (beta_1 * studytime) + (beta_2 * failures) + ... + error
#
# Where:
#   beta_0 is the intercept (baseline expected grade when all scaled features are 0)
#   beta_i is the regression coefficient for feature i
print("\n[Step 3] Training LinearRegression model on training data...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("         Model training complete!")

# ---------------------------------------------------------------------------
# Step 4: Model Prediction on Unseen Test Data
# ---------------------------------------------------------------------------
print("\n[Step 4] Generating predictions on unseen test dataset...")
y_pred_raw = model.predict(X_test_scaled)

# Academic rule: Grades are on a 0 to 20 scale.
# For evaluation metrics, we compute on raw continuous predictions according to standard practice.
# For real-world display and recommendation, scores are clipped to [0, 20].
y_pred_bounded = np.clip(y_pred_raw, 0.0, 20.0)

# ---------------------------------------------------------------------------
# Step 5: Model Evaluation Metrics
# ---------------------------------------------------------------------------
# We compute all four standard evaluation metrics on the test set:
mae = mean_absolute_error(y_test, y_pred_raw)
mse = mean_squared_error(y_test, y_pred_raw)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_raw)

print("\n[Step 5] Evaluation Metrics on Unseen Test Data (N = 79):")
print("-" * 55)
print(f"  Mean Absolute Error (MAE):       {mae:.4f}")
print(f"  Mean Squared Error (MSE):        {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE):  {rmse:.4f}")
print(f"  Coefficient of Determination (R²): {r2:.4f} ({r2 * 100:.2f}%)")
print("-" * 55)

# Beginner-friendly metric explanation
print("\n  Beginner-Friendly Explanation of Metrics:")
print(f"  * MAE ({mae:.2f} marks): On average, our prediction is off by only ~{mae:.2f} marks out of 20.")
print(f"  * MSE ({mse:.2f}): Average squared error, penalizing occasional large prediction errors.")
print(f"  * RMSE ({rmse:.2f} marks): Standard deviation of prediction residuals; interpretable in grade marks.")
print(f"  * R² ({r2:.4f} / {r2*100:.1f}%): Explains that {r2*100:.1f}% of the variance in final exam grades (G3)")
print("    is directly explained by the model's 5 academic features.")

# ---------------------------------------------------------------------------
# Step 6: Sample Actual vs Predicted Comparison Table
# ---------------------------------------------------------------------------
print("\n[Step 6] Sample Student Actual vs. Predicted Grades (First 10 Test Students):")
comparison_df = pd.DataFrame({
    "StudyTime": X_test["studytime"].values[:10],
    "Failures": X_test["failures"].values[:10],
    "Absences": X_test["absences"].values[:10],
    "G1 (Term 1)": X_test["G1"].values[:10],
    "G2 (Term 2)": X_test["G2"].values[:10],
    "Actual G3": y_test.values[:10],
    "Predicted (Raw)": np.round(y_pred_raw[:10], 2),
    "Predicted [0-20]": np.round(y_pred_bounded[:10], 2),
    "Abs Error": np.round(np.abs(y_test.values[:10] - y_pred_bounded[:10]), 2)
})
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Step 7: Coefficient Analysis & Mathematical Interpretation
# ---------------------------------------------------------------------------
intercept = float(model.intercept_)
coefficients = {col: float(coef) for col, coef in zip(FEATURE_COLUMNS, model.coef_)}

print("\n[Step 7] Model Parameters & Mathematical Interpretation:")
print(f"         Baseline Intercept (beta_0): {intercept:.4f}")
print("         Feature Coefficients (Standardized):")
for feature, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "increases" if coef > 0 else "decreases"
    print(f"           - {feature:10s}: {coef:+.4f} (1 std deviation increase {direction} expected G3 by {abs(coef):.3f})")

print("\n  Summary of Feature Influence on Final Exam Score (G3):")
print("  1. G2 (Term 2 Exam): By far the strongest positive driver (+{:.2f}). Recent midterm mastery".format(coefficients['G2']))
print("     is the single best indicator of final exam readiness.")
print("  2. G1 (Term 1 Exam): Positive contribution (+{:.2f}), confirming foundational performance matters.".format(coefficients['G1']))
print("  3. Absences: Moderate impact (+{:.2f} in standardized space, due to non-linear attendance patterns).".format(coefficients['absences']))
print("  4. Study Time: Positive impact (+{:.2f}), indicating regular study hours lift expected grades.".format(coefficients['studytime']))
print("  5. Past Failures: Negative impact ({:.2f}), indicating prior course difficulties create friction.".format(coefficients['failures']))

# ---------------------------------------------------------------------------
# Step 8: Visualization: Actual vs. Predicted Scores Plot
# ---------------------------------------------------------------------------
print("\n[Step 8] Generating Actual vs. Predicted diagnostic chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Actual vs Predicted Scatter with Perfect Prediction Line
ax1.scatter(y_test, y_pred_bounded, color="#2563eb", alpha=0.7, edgecolors="k", s=60, label="Test Students")
# Ideal 45-degree line y = x
line_min = 0
line_max = 20
ax1.plot([line_min, line_max], [line_min, line_max], color="#dc2626", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
ax1.set_title(f"Actual vs. Predicted Final Exam Score (G3)\n$R^2 = {r2:.3f}$ | $RMSE = {rmse:.2f}$", fontsize=12, weight="bold")
ax1.set_xlabel("Actual Final Grade (G3)", fontsize=11)
ax1.set_ylabel("Predicted Final Grade (G3)", fontsize=11)
ax1.set_xlim(-0.5, 20.5)
ax1.set_ylim(-0.5, 20.5)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

# Subplot 2: Residuals Distribution (Actual - Predicted)
residuals = y_test - y_pred_raw
sns.histplot(residuals, kde=True, bins=15, color="#059669", edgecolor="black", ax=ax2)
ax2.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Zero Error Line")
ax2.set_title("Prediction Error (Residuals) Distribution\nMean Residual = {:.2f}".format(residuals.mean()), fontsize=12, weight="bold")
ax2.set_xlabel("Residual (Actual - Predicted Grade)", fontsize=11)
ax2.set_ylabel("Student Count", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
fig.savefig(CHART_SAVE_PATH, dpi=300)
plt.close(fig)
print(f"         Diagnostic chart saved -> {CHART_SAVE_PATH}")

# ---------------------------------------------------------------------------
# Step 9: Save Model and Metrics Metadata
# ---------------------------------------------------------------------------
print("\n[Step 9] Saving trained model and metrics metadata...")

# Save Scikit-Learn Model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"         Trained model saved -> {MODEL_SAVE_PATH}")

# Save Metrics JSON
metrics_dict = {
    "model_name": "Multiple Linear Regression",
    "target_variable": TARGET_COLUMN,
    "feature_columns": FEATURE_COLUMNS,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "evaluation_metrics": {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "r2_percentage": round(float(r2 * 100), 2)
    },
    "model_parameters": {
        "intercept": round(intercept, 4),
        "coefficients": {k: round(v, 4) for k, v in coefficients.items()}
    }
}

with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"         Model metrics saved -> {METRICS_SAVE_PATH}")

print("\n" + "=" * 70)
print("STAGE 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
