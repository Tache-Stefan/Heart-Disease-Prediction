import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from feature_engineering import create_features

create_features()
df = pd.read_csv("../data/improved_dataset.csv")
x = df.drop(columns=["heart_disease"])
y = df["heart_disease"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred) * 100
roc_auc = roc_auc_score(y_test, y_prob) * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"ROC AUC: {roc_auc:.2f}%")

joblib.dump(model, "../models/xgb_heart_disease_model.pkl")
print("Model saved to models/xgb_heart_disease_model.pkl")
