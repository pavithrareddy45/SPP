"""
=============================================================================
SQLite Database Manager for Prediction Records
=============================================================================
Module: src/database.py
Project: Student Performance Predictor

Handles SQLite persistence for successful student predictions:
- Stores timestamp, model inputs, predicted grade, performance tier, momentum.
- Strictly excludes sensitive personal identifiers (no names, emails, IDs).
- Computes dynamic cohort-level dashboard statistics.
- Gracefully handles empty database states.
=============================================================================
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "predictions.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection and configures row factory for dict-like access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the predictions table if it does not already exist."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                studytime INTEGER NOT NULL,
                failures INTEGER NOT NULL,
                absences INTEGER NOT NULL,
                g1 REAL NOT NULL,
                g2 REAL NOT NULL,
                predicted_score REAL NOT NULL,
                performance_tier TEXT NOT NULL,
                grade_momentum TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_prediction(
    studytime: int,
    failures: int,
    absences: int,
    g1: float,
    g2: float,
    predicted_score: float,
    performance_tier: str,
    grade_momentum: str,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Saves a successful prediction record into SQLite.
    Returns the inserted record ID.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                prediction_date, studytime, failures, absences,
                g1, g2, predicted_score, performance_tier, grade_momentum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, studytime, failures, absences,
            float(g1), float(g2), float(predicted_score),
            str(performance_tier), str(grade_momentum)
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_predictions(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Returns all prediction records ordered by most recent first."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_dashboard_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Computes cohort statistics dynamically from the predictions table.
    Returns structured stats, handling empty database states gracefully.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. Total Count
        cursor.execute("SELECT COUNT(*) AS total FROM predictions")
        total_students = cursor.fetchone()["total"]

        if total_students == 0:
            return {
                "total_students": 0,
                "avg_predicted_grade": 0.0,
                "pass_rate": 0.0,
                "at_risk_count": 0,
                "pass_count": 0,
                "merit_count": 0,
                "distinction_count": 0,
                "has_data": False
            }

        # 2. Average predicted grade
        cursor.execute("SELECT AVG(predicted_score) AS avg_score FROM predictions")
        avg_score = round(float(cursor.fetchone()["avg_score"]), 2)

        # 3. Passing count (score >= 10.0)
        cursor.execute("SELECT COUNT(*) AS pass_cnt FROM predictions WHERE predicted_score >= 10.0")
        pass_count = cursor.fetchone()["pass_cnt"]
        pass_rate = round((pass_count / total_students) * 100, 1)

        # 4. At-risk count (score < 10.0)
        cursor.execute("SELECT COUNT(*) AS risk_cnt FROM predictions WHERE predicted_score < 10.0")
        at_risk_count = cursor.fetchone()["risk_cnt"]

        # 5. Tier counts for breakdown
        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE performance_tier = 'Distinction'")
        distinction_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE performance_tier = 'Merit'")
        merit_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE performance_tier = 'Pass'")
        standard_pass_count = cursor.fetchone()["cnt"]

        return {
            "total_students": total_students,
            "avg_predicted_grade": avg_score,
            "pass_rate": pass_rate,
            "at_risk_count": at_risk_count,
            "pass_count": standard_pass_count,
            "merit_count": merit_count,
            "distinction_count": distinction_count,
            "has_data": True
        }
    finally:
        conn.close()


def clear_all_predictions(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clears all records from the predictions table (used for testing)."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()
