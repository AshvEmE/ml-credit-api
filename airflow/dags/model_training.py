import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def main():
    print("Навчання моделі...")


    data_path = "/opt/airflow/data/credit_data.csv"
    model_path = "/opt/airflow/model.pkl"

    df = pd.read_csv(data_path)

    X = df.drop("approved", axis=1)
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Модель збережено у '{model_path}'")

if __name__ == "__main__":
    main()
