from fastapi import FastAPI
from pydantic import BaseModel  # Додаємо імпорт BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from fastapi.responses import HTMLResponse


app = FastAPI(title="Credit Approval API")

model_path = "model.pkl"


# Завантаження моделі
def load_model():
    return joblib.load(model_path)


model = load_model()


# Схема запиту на передбачення
class CreditApplication(BaseModel):  # Використовуємо BaseModel для валідації даних
    age: int
    income: float
    loan_amount: float
    loan_term: int
    employment_years: int
    credit_score: int
    has_other_loans: int  # 0 або 1


@app.get("/")
async def root():
    try:
        # Шлях до HTML файлу
        file_path = os.path.join(os.getcwd(), "frontend", "credit_form.html")

        # Відкриваємо файл з кодуванням UTF-8
        with open(file_path, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        # Повертаємо помилку, якщо файл не вдалося відкрити
        return {"error": f"Failed to load file: {str(e)}"}


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


# Переобучення моделі
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

    return {"status": "Модель переобучено успішно"}
