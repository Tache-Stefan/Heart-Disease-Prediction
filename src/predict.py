import joblib


def predict_heart_risk(df, model_path="../models/xgb_heart_disease_model.pkl"):
    model = joblib.load(model_path)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    return predictions, probabilities
