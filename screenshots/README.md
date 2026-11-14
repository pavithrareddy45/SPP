# Screenshots Guide for Project Presentation & Faculty Evaluation

This document outlines the required screenshots to capture for your academic report, project submission, and presentation slides.

---

## 1. Required Screenshots Checklist

| # | Screenshot Name | Recommended File Name | Page URL / Source | What to Show |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Home Page & Hero Banner** | `01_home_page.png` | `http://127.0.0.1:5000/` | Top navigation bar with brand icon, hero title, and quick-test preset buttons (🌸 Try High Achiever, 🍂 Try At-Risk Student). |
| **2** | **Student Prediction Form** | `02_prediction_form.png` | `http://127.0.0.1:5000/` | Filled input fields (Weekly Study Hours, Past Failures, Absences, G1, and G2) alongside the Model Reliability sidebar ($R^2=78.2\%$, $\text{MAE}=1.34$). |
| **3** | **Prediction Results & Action Plan** | `03_prediction_results.png` | `http://127.0.0.1:5000/predict` | The Pinterest-style score circle ($18.3/20$), Performance Tier badge, Grade Momentum pill, and personalized 3-phase improvement roadmap. |
| **4** | **Teacher Cohort Dashboard** | `04_teacher_dashboard.png` | `http://127.0.0.1:5000/dashboard` | The 4 dynamic summary cards (Total Students Analyzed, Class Average Score, Pass Rate %, At-Risk Count) and the filterable student summary table. |
| **5** | **Visual Analytics Charts** | `05_analytics_charts.png` | `http://127.0.0.1:5000/dashboard#analytics` | The gallery of diagnostic charts: Final Grade Distribution, Correlation Heatmap, Study Time vs. G3, Absences vs. G3, and Actual vs. Predicted regression plot. |

---

## 2. Step-by-Step Instructions to Capture

1. **Start the Flask Application:**
   ```powershell
   py app.py
   ```
2. **Open your web browser:**
   Go to `http://127.0.0.1:5000/`.
3. **Capture Screenshot 1 (Home Page):**
   * Use the Windows Snipping Tool (`Win + Shift + S`) or browser full-page capture.
   * Save as `screenshots/01_home_page.png`.
4. **Capture Screenshot 2 (Prediction Form):**
   * Click **"🌸 Try High Achiever"** to fill sample inputs.
   * Capture the filled form and sidebar card.
   * Save as `screenshots/02_prediction_form.png`.
5. **Capture Screenshot 3 (Prediction Results):**
   * Click **"✨ Predict Final Grade & Get Advisory Plan"**.
   * Capture the resulting score badge, tier pill, and personalized advisory cards.
   * Save as `screenshots/03_prediction_results.png`.
6. **Capture Screenshot 4 & 5 (Teacher Dashboard & Charts):**
   * Navigate to `http://127.0.0.1:5000/dashboard`.
   * Capture the 4 summary statistics cards and the student records table.
   * Scroll down and capture the empirical visual analytics charts.
   * Save as `screenshots/04_teacher_dashboard.png` and `screenshots/05_analytics_charts.png`.
