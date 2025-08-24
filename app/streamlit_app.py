import os
import sys
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_heart_disease_model.pkl")
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from feature_engineering import create_features

model = joblib.load(MODEL_PATH)

st.title("Heart Disease Risk Prediction")
st.write("Enter patient information to predict the risk of heart disease.")

# --- Inputs ---
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [3, 6, 7])
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", [0, 1])
diabetes = st.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
bmi = st.number_input("BMI", 15.0, 45.0, 25.0, step=0.1)

df = pd.DataFrame(
    [
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "smoking": smoking,
            "diabetes": diabetes,
            "bmi": bmi,
        }
    ]
)

df = create_features(df)
model_columns = model.get_booster().feature_names
df = df.reindex(columns=model_columns, fill_value=0)

if st.button("Predict"):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease (probability {probability:.2f})")
    else:
        st.success(f"Low risk of heart disease (probability {probability:.2f})")

    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.codes

    st.subheader("Explanation of Prediction")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    st.write("Top factors influencing this prediction:")
    shap.initjs()
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    st.pyplot(fig)
