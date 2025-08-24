# Heart Disease Risk Prediction  

A machine learning project that predicts the risk of heart disease based on patient health indicators.  
Built with **XGBoost**, feature engineering, and an interactive **Streamlit web app** with explainability (SHAP).  

---

## Project Overview  

This project demonstrates how **machine learning** can help assess patient risk factors and provide interpretable insights for medical decision support.  

Key highlights:  
- **Exploratory Data Analysis (EDA)** and **feature engineering**  
- **XGBoost model** with hyperparameter tuning and SMOTE balancing  
- **Explainability** using SHAP
- **Interactive web app** built with Streamlit for user-friendly predictions  

---

## 🚀 How to Run  

### 1. Clone the repo  
```bash
git clone https://github.com/Tache-Stefan/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```
### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## Model Performance

- **Algorithm**: XGBoost
- **Handling imbalance**: SMOTE oversampling
- **Best metrics (test set)**:
  - Accuracy: ~54%
  - ROC AUC: ~52%
  - Stronger recall on high-risk patients compared to baseline
