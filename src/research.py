import os
import sqlite3
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "database", "history.db")


def load_predictions_dataframe():
    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql_query("""
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
        ORDER BY id ASC
    """, conn)

    conn.close()

    if not df.empty:
        df["analysis_date"] = pd.to_datetime(df["analysis_date"])

    return df


def get_class_distribution():
    df = load_predictions_dataframe()

    if df.empty:
        return {}

    return df["predicted_class"].value_counts().to_dict()


def get_average_confidence_by_class():
    df = load_predictions_dataframe()

    if df.empty:
        return {}

    return df.groupby("predicted_class")["confidence"].mean().round(2).to_dict()


def get_analysis_count_by_date():
    df = load_predictions_dataframe()

    if df.empty:
        return {}

    df["date"] = df["analysis_date"].dt.date
    return df.groupby("date").size().to_dict()


def get_total_analyzed():
    df = load_predictions_dataframe()
    return len(df)


if __name__ == "__main__":
    print("Загальна кількість аналізів:", get_total_analyzed())
    print("Розподіл по класах:", get_class_distribution())
    print("Середня впевненість:", get_average_confidence_by_class())
    print("Кількість аналізів по датах:", get_analysis_count_by_date())