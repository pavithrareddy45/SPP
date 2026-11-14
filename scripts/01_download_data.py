"""
Stage 1: Dataset Sourcing, Inspection, and Ethical Feature Selection.

Author: Senior Python & ML Developer
Project: Student Performance Predictor
Source: UCI Machine Learning Repository (ID: 320)
Citation: Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary
          School Student Performance. In Proceedings of 5th Annual Future
          Business Technology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal.
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
ZIP_PATH = os.path.join(RAW_DATA_DIR, "student_performance.zip")
CSV_PATH = os.path.join(RAW_DATA_DIR, "student-mat.csv")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "student_performance_clean.csv")


def download_dataset():
    """Downloads the official UCI Student Performance dataset zip file."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    print(f"[1/5] Downloading dataset from UCI repository:\n      {UCI_URL}")
    if not os.path.exists(ZIP_PATH):
        urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
        print("      Download complete!")
    else:
        print("      Zip archive already exists locally. Skipping download.")

    print("[2/5] Extracting dataset files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    # The UCI archive contains a nested student.zip
    nested_zip = os.path.join(RAW_DATA_DIR, "student.zip")
    if os.path.exists(nested_zip):
        with zipfile.ZipFile(nested_zip, 'r') as nested_ref:
            nested_ref.extractall(RAW_DATA_DIR)
        print("      Extracted nested student.zip contents to:", RAW_DATA_DIR)
    else:
        print("      Extracted contents to:", RAW_DATA_DIR)


def inspect_raw_data():
    """Loads raw CSV and prints summary statistics."""
    print("[3/5] Inspecting raw dataset (student-mat.csv)...")
    # UCI dataset uses ';' delimiter
    df_raw = pd.read_csv(CSV_PATH, sep=';')
    print(f"      Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"      Raw Columns: {list(df_raw.columns)}")
    return df_raw


def filter_ethical_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ethical AI guidelines:
    Removes sensitive demographic, socio-economic, and personal attributes.
    Retains only academic and study-behavioral predictors.
    """
    print("[4/5] Applying Ethical Feature Selection...")
    
    # Selected purely academic and behavioral features
    academic_features = [
        "studytime",   # Weekly study hours (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)
        "failures",    # Number of past class failures (0 to 4)
        "absences",    # Number of school absences (0 to 93)
        "G1",          # First period assessment/assignment grade (0 to 20)
        "G2",          # Second period examination grade (0 to 20)
        "G3"           # Final exam score (0 to 20) - Target variable
    ]

    # Excluded features for transparency
    excluded_features = [col for col in df_raw.columns if col not in academic_features]
    print(f"      Excluded {len(excluded_features)} sensitive/non-academic columns:")
    print(f"      {excluded_features}")

    df_clean = df_raw[academic_features].copy()

    # Data verification
    print("\n[5/5] Verifying Cleaned Dataset Integrity...")
    print(f"      Cleaned Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"      Missing Values Check:\n{df_clean.isnull().sum().to_string()}")
    print("\n      Descriptive Statistics:")
    print(df_clean.describe().round(2).to_string())

    # Save to processed data directory
    df_clean.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"\n[OK] Cleaned dataset saved to:\n     {PROCESSED_CSV_PATH}")

    return df_clean


def generate_data_documentation(df_clean: pd.DataFrame):
    """Generates a Markdown documentation file explaining the dataset provenance and columns."""
    doc_content = f"""# Dataset Documentation: Student Performance Predictor

## 1. Provenance & Citation
- **Source:** [UCI Machine Learning Repository - Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Original Authors:** Paulo Cortez and Alice Silva (University of Minho, Portugal)
- **Publication:** Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal. ISBN: 978-9077381-39-7.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Ethical Machine Learning Rationale
To comply with ethical AI principles and prevent algorithmic bias:
- **Zero Demographic Bias:** Personal demographic attributes (`sex`, `age`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `guardian`, `romantic`, etc.) were intentionally **excluded** from the model.
- **Academic Focus:** The predictor relies purely on measurable academic indicators, attendance, and study habits.

---

## 3. Data Dictionary

| Column Name | Type | Allowed Range / Scale | Description | Role in ML |
| :--- | :--- | :--- | :--- | :--- |
| `studytime` | Categorical / Ordinal | 1 to 4 | Weekly study hours:<br>• 1: `<2 hours`<br>• 2: `2 to 5 hours`<br>• 3: `5 to 10 hours`<br>• 4: `>10 hours` | Feature (Predictor) |
| `failures` | Integer | 0 to 4 | Number of past class failures (0 if none, up to 4) | Feature (Predictor) |
| `absences` | Integer | 0 to 93 | Number of school absences across the school term | Feature (Predictor) |
| `G1` | Numeric / Float | 0 to 20 | First period evaluation / assignment grade | Feature (Predictor) |
| `G2` | Numeric / Float | 0 to 20 | Second period midterm exam score | Feature (Predictor) |
| `G3` | Numeric / Float | 0 to 20 | Final examination grade | **Target Variable ($y$)** |

---

## 4. Dataset Summary & Health
- **Total Records:** {df_clean.shape[0]}
- **Total Features:** {df_clean.shape[1] - 1} predictors + 1 target
- **Missing / Null Values:** 0 (Dataset is complete and clean)
- **Target Variable ($G3$) Mean:** {df_clean['G3'].mean():.2f} (Std: {df_clean['G3'].std():.2f}, Min: {df_clean['G3'].min()}, Max: {df_clean['G3'].max()})
"""
    doc_path = os.path.join(DOCS_DIR, "dataset_documentation.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    print(f"[OK] Data documentation created at:\n     {doc_path}")


if __name__ == "__main__":
    try:
        download_dataset()
        raw_df = inspect_raw_data()
        clean_df = filter_ethical_features(raw_df)
        generate_data_documentation(clean_df)
        print("\n=== Stage 1 Completed Successfully! ===")
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}", file=sys.stderr)
        sys.exit(1)
