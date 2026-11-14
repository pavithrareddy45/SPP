# Dataset Documentation: Student Performance Predictor

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
- **Total Records:** 395
- **Total Features:** 5 predictors + 1 target
- **Missing / Null Values:** 0 (Dataset is complete and clean)
- **Target Variable ($G3$) Mean:** 10.42 (Std: 4.58, Min: 0, Max: 20)
