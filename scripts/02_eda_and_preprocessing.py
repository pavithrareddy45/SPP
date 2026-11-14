"""
=============================================================================
STAGE 2: Exploratory Data Analysis (EDA) & Leak-Free Preprocessing Pipeline
=============================================================================
Project: Student Performance Predictor
Target Variable: G3 (Final Examination Grade, 0-20 scale)

Key Objectives of this Stage:
1. Conduct Exploratory Data Analysis (EDA) on academic features.
2. Generate and save 4 key diagnostic charts to 'static/images/':
   - Final Grade Distribution (G3)
   - Correlation Heatmap (All features vs G3)
   - Study Time vs G3 (Boxplot by study hours category)
   - Absences vs G3 (Scatter plot with trendline)
3. Implement a Leak-Free Train-Test Split (80% train, 20% test, random_state=42).
4. Strictly separate input features (X) from target (y), ensuring G3 is never in X.
5. Construct and fit a Scikit-learn preprocessing pipeline solely on training data.
6. Print dataset shapes and verify zero data leakage.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------------------------
# Step 1: Directory Setup and Data Loading
# ---------------------------------------------------------------------------
# We locate the project root directory dynamically so the script runs cleanly
# regardless of where it is invoked from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "student_performance_clean.csv")

print("[Step 1] Loading cleaned dataset from:")
print(f"         {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"         Successfully loaded dataset: {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Set visualization aesthetics for clean, publication-ready figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})

# ---------------------------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA) & Chart Generation
# ---------------------------------------------------------------------------
print("[Step 2] Generating Exploratory Data Analysis (EDA) figures...")

# ---------------------------------------------------------------------------
# Chart 1: Final Grade (G3) Distribution
# ---------------------------------------------------------------------------
# What this shows:
# How students' final marks are spread across the 0 to 20 range.
# In Portuguese secondary schools (the dataset origin), the passing grade is 10/20.
# A vertical line at x=10 visually separates passing from failing scores.
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df["G3"], kde=True, bins=21, color="#2563eb", edgecolor="black", ax=ax1)
ax1.axvline(10, color="#dc2626", linestyle="--", linewidth=2, label="Passing Threshold (10/20)")
ax1.set_title("Distribution of Final Exam Grades (G3)", fontsize=14, weight="bold", pad=12)
ax1.set_xlabel("Final Exam Score (0 to 20)", fontsize=12)
ax1.set_ylabel("Student Count", fontsize=12)
ax1.set_xlim(-0.5, 20.5)
ax1.legend(loc="upper right", frameon=True)
plt.tight_layout()

chart1_path = os.path.join(IMAGES_DIR, "final_grade_distribution.png")
fig1.savefig(chart1_path, dpi=300)
plt.close(fig1)
print(f"   [OK] Saved Chart 1 -> {chart1_path}")

# ---------------------------------------------------------------------------
# Chart 2: Correlation Heatmap
# ---------------------------------------------------------------------------
# What this shows:
# Pearson correlation coefficient between each feature and G3 (target).
# Values range from -1 (strong inverse relationship) to +1 (strong direct relationship).
# G1 and G2 have strong positive correlations with G3, while absences and failures
# show negative correlations.
fig2, ax2 = plt.subplots(figsize=(7, 6))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Optional: mask upper triangle for readability
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar_kws={"label": "Pearson Correlation"},
    linewidths=0.5,
    ax=ax2
)
ax2.set_title("Feature Correlation Heatmap", fontsize=14, weight="bold", pad=12)
plt.tight_layout()

chart2_path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
fig2.savefig(chart2_path, dpi=300)
plt.close(fig2)
print(f"   [OK] Saved Chart 2 -> {chart2_path}")

# ---------------------------------------------------------------------------
# Chart 3: Study Time vs Final Grade (G3)
# ---------------------------------------------------------------------------
# What this shows:
# How weekly study time correlates with academic achievement.
# Studytime is mapped to clear, intuitive hour categories:
# 1 = <2 hours/week, 2 = 2-5 hours/week, 3 = 5-10 hours/week, 4 = >10 hours/week.
study_labels = {1: "< 2 hrs/wk", 2: "2 - 5 hrs/wk", 3: "5 - 10 hrs/wk", 4: "> 10 hrs/wk"}
df_plot = df.copy()
df_plot["studytime_label"] = df_plot["studytime"].map(study_labels)

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_plot,
    x="studytime_label",
    y="G3",
    hue="studytime_label",
    legend=False,
    palette="Blues",
    order=["< 2 hrs/wk", "2 - 5 hrs/wk", "5 - 10 hrs/wk", "> 10 hrs/wk"],
    ax=ax3
)
sns.stripplot(
    data=df_plot,
    x="studytime_label",
    y="G3",
    color="#1e293b",
    alpha=0.3,
    jitter=0.2,
    order=["< 2 hrs/wk", "2 - 5 hrs/wk", "5 - 10 hrs/wk", "> 10 hrs/wk"],
    ax=ax3
)
ax3.axhline(10, color="#dc2626", linestyle=":", linewidth=1.5, label="Passing Mark (10)")
ax3.set_title("Weekly Study Time vs. Final Exam Score (G3)", fontsize=14, weight="bold", pad=12)
ax3.set_xlabel("Weekly Study Time Category", fontsize=12)
ax3.set_ylabel("Final Exam Score (G3)", fontsize=12)
ax3.legend(loc="lower right")
plt.tight_layout()

chart3_path = os.path.join(IMAGES_DIR, "study_time_vs_g3.png")
fig3.savefig(chart3_path, dpi=300)
plt.close(fig3)
print(f"   [OK] Saved Chart 3 -> {chart3_path}")

# ---------------------------------------------------------------------------
# Chart 4: School Absences vs Final Grade (G3)
# ---------------------------------------------------------------------------
# What this shows:
# The relationship between class attendance/absenteeism and final exam outcomes.
# We include a linear trendline (regression line) to visualize the trajectory.
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.regplot(
    data=df,
    x="absences",
    y="G3",
    scatter_kws={"alpha": 0.5, "color": "#2563eb"},
    line_kws={"color": "#dc2626", "linewidth": 2},
    ax=ax4
)
ax4.set_title("School Absences vs. Final Exam Score (G3)", fontsize=14, weight="bold", pad=12)
ax4.set_xlabel("Number of Absences During Term", fontsize=12)
ax4.set_ylabel("Final Exam Score (G3)", fontsize=12)
plt.tight_layout()

chart4_path = os.path.join(IMAGES_DIR, "absences_vs_g3.png")
fig4.savefig(chart4_path, dpi=300)
plt.close(fig4)
print(f"   [OK] Saved Chart 4 -> {chart4_path}\n")

# ---------------------------------------------------------------------------
# Step 3: Feature & Target Separation (Avoiding Target Contamination)
# ---------------------------------------------------------------------------
print("[Step 3] Separating Features (X) and Target (y)...")

# Target variable: G3
TARGET_COLUMN = "G3"
y = df[TARGET_COLUMN].copy()

# Features: All columns EXCEPT G3
# This strictly guarantees that G3 is NEVER present in the input feature matrix X.
FEATURE_COLUMNS = ["studytime", "failures", "absences", "G1", "G2"]
X = df[FEATURE_COLUMNS].copy()

print(f"         Target variable (y): '{TARGET_COLUMN}'")
print(f"         Input features (X):  {FEATURE_COLUMNS}")
print(f"         Verification: Is '{TARGET_COLUMN}' inside X? {'YES (Error!)' if TARGET_COLUMN in X.columns else 'NO (Correct!)'}\n")

# ---------------------------------------------------------------------------
# Step 4: Leak-Free Train-Test Split (80% Train, 20% Test)
# ---------------------------------------------------------------------------
# WHY WE SPLIT BEFORE PREPROCESSING (Avoiding Data Leakage):
# In real-world machine learning, the test set represents future, unseen students.
# If we calculate statistics (e.g., mean or standard deviation for scaling) using
# the entire dataset, information from the test set 'leaks' into the training
# process. This causes overly optimistic evaluation results.
# By splitting FIRST, all preprocessing transformations learn ONLY from X_train.
print("[Step 4] Performing Leak-Free Train-Test Split (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"         Training Set (X_train): {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"         Testing Set  (X_test):  {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"         Training Target (y_train): {y_train.shape[0]} samples")
print(f"         Testing Target  (y_test):  {y_test.shape[0]} samples\n")

# ---------------------------------------------------------------------------
# Step 5: Scikit-learn Preprocessing Pipeline Construction
# ---------------------------------------------------------------------------
# We use StandardScaler to normalize features to mean=0 and variance=1.
# This puts features on a common scale (e.g. absences 0-75 vs studytime 1-4),
# making linear regression coefficients directly comparable and interpretable.
#
# RULE APPLIED: .fit() is ONLY called on X_train.
# X_test is transformed using .transform() using X_train's learned statistics.
print("[Step 5] Fitting Preprocessing Pipeline (StandardScaler) on Training Data ONLY...")

preprocessor_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Fit ONLY on training features
preprocessor_pipeline.fit(X_train)

# Transform training and testing features independently
X_train_scaled = preprocessor_pipeline.transform(X_train)
X_test_scaled = preprocessor_pipeline.transform(X_test)

# Verify statistics learned solely from X_train
scaler_step = preprocessor_pipeline.named_steps["scaler"]
print("         Learned Means from X_train:")
for feat, mean_val in zip(FEATURE_COLUMNS, scaler_step.mean_):
    print(f"           - {feat:10s}: {mean_val:.2f}")

# ---------------------------------------------------------------------------
# Step 6: Persisting Preprocessed Data & Pipeline for Stage 3
# ---------------------------------------------------------------------------
print("\n[Step 6] Saving Split Datasets & Preprocessing Pipeline for Stage 3...")

# Save train and test data frames to disk
train_df = X_train.copy()
train_df[TARGET_COLUMN] = y_train
train_csv_path = os.path.join(PROCESSED_DATA_DIR, "train_data.csv")
train_df.to_csv(train_csv_path, index=False)

test_df = X_test.copy()
test_df[TARGET_COLUMN] = y_test
test_csv_path = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")
test_df.to_csv(test_csv_path, index=False)

# Save the fitted preprocessing pipeline
pipeline_path = os.path.join(MODELS_DIR, "preprocessor_pipeline.pkl")
joblib.dump(preprocessor_pipeline, pipeline_path)

print(f"         Train Data Saved -> {train_csv_path}")
print(f"         Test Data Saved  -> {test_csv_path}")
print(f"         Pipeline Saved   -> {pipeline_path}")

print("\n" + "=" * 70)
print("STAGE 2 VERIFICATION SUMMARY:")
print("=" * 70)
print(f"  Target Variable:          {TARGET_COLUMN}")
print(f"  Feature Variables:        {FEATURE_COLUMNS}")
print(f"  Total Samples:            {len(df)}")
print(f"  Training Samples (80%):   {X_train.shape[0]} (Shape: {X_train.shape})")
print(f"  Testing Samples  (20%):   {X_test.shape[0]} (Shape: {X_test.shape})")
print(f"  Target Included in X?     NO - Strictly separated")
print(f"  Data Leakage Prevented?   YES - Preprocessor fit only on X_train")
print(f"  Charts Generated:         4 figures saved in 'static/images/'")
print(f"  Ready for Stage 3 Model Training? YES")
print("=" * 70)
