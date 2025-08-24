import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    output_path = os.path.join(BASE_DIR, "data", "improved_dataset.csv")

    df["age_group"] = pd.cut(
        df["age"], bins=[28, 40, 50, 60, 70, 80], labels=[0, 1, 2, 3, 4]
    )
    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["bp_high"] = (df["trestbps"] > 140).astype(int)
    df["hr_reserve"] = df["thalach"] - df["age"]
    df["risk_factors"] = df["smoking"] + df["diabetes"] + df["fbs"] + df["exang"]
    df["st_risk"] = df["slope"] * df["oldpeak"]
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 35, 50], labels=[0, 1, 2, 3, 4]
    )
    df["age_x_chol"] = df["age"] * df["chol"]
    df["bmi_x_chol"] = df["bmi"] * df["chol"]
    df["thalach_x_age"] = df["thalach"] * df["age"]
    df["chol_high"] = (df["chol"] > 240).astype(int)
    df["chol_low"] = (df["chol"] < 160).astype(int)
    df["max_hr_predicted"] = 220 - df["age"]
    df["hr_percent"] = df["thalach"] / df["max_hr_predicted"]
    df["weighted_risk"] = (
        2 * df["diabetes"] + 2 * df["smoking"] + df["fbs"] + df["exang"]
    )
    df = pd.get_dummies(df, columns=["cp", "thal", "slope", "restecg"], drop_first=True)

    df.to_csv(output_path, index=False)
    return df
