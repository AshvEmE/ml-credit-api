from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

app = FastAPI(title="Credit Approval API")

model_path = "model.pkl"

#  Завантаження моделі
def load_model():
    return joblib.load(model_path)

model = load_model()

#  Схема запиту на передбачення
class CreditApplication(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_term: int
    employment_years: int
    credit_score: int
    has_other_loans: int  # 0 або 1

@app.post("/predict")
def predict(applicant: CreditApplication):
    features = np.array([
        applicant.age,
        applicant.income,
        applicant.loan_amount,
        applicant.loan_term,
        applicant.employment_years,
        applicant.credit_score,
        applicant.has_other_loans
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    return {
        "approved": bool(prediction),
        "message": "✅ Credit Approved" if prediction else "❌ Credit Denied"
    }

#  Переобучення моделі
@app.post("/retrain")
def retrain():
    file_path = "data/credit_data.csv"
    if not os.path.exists(file_path):
        return {"error": "Файл 'data/credit_data.csv' не знайдено"}

    df = pd.read_csv(file_path)

    if "approved" not in df.columns:
        return {"error": "CSV має містити стовпець 'approved'"}

    X = df.drop("approved", axis=1)
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_model = RandomForestClassifier()
    new_model.fit(X_train, y_train)

    joblib.dump(new_model, model_path)
    global model
    model = load_model()  # перезавантажуємо модель у пам’ять

    return {"status": " Модель переобучено успішно"}
