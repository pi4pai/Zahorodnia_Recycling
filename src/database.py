import os
import sqlite3
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "history.db")

os.makedirs(DATABASE_DIR, exist_ok=True)


def create_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_30_50 REAL NOT NULL,
            prob_50_70 REAL NOT NULL,
            prob_70_100 REAL NOT NULL,
            recycling_method TEXT NOT NULL,
            recycling_description TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(result):
    create_database()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    probabilities = result["probabilities"]

    cursor.execute("""
        INSERT INTO predictions (
            analysis_date,
            image_path,
            predicted_class,
            confidence,
            prob_30_50,
            prob_50_70,
            prob_70_100,
            recycling_method,
            recycling_description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result["image_path"],
        result["predicted_class"],
        result["confidence"],
        probabilities.get("30-50", 0),
        probabilities.get("50-70", 0),
        probabilities.get("70-100", 0),
        result["recycling_method"],
        result["recycling_description"]
    ))

    conn.commit()
    conn.close()


def get_all_predictions():
    create_database()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            id,
            analysis_date,
            image_path,
            predicted_class,
            confidence,
            prob_30_50,
            prob_50_70,
            prob_70_100,
            recycling_method
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows


def clear_history():
    create_database()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()