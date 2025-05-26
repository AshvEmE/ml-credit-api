import pandas as pd
import random
import os

def generate_applicant():
    age = random.randint(18, 70)
    income = random.randint(8000, 150000)
    loan_amount = random.randint(1000, 70000)
    loan_term = random.choice([6, 12, 24, 36, 48, 60])
    employment_years = random.randint(0, age - 18)
    credit_score = random.randint(300, 850)
    has_other_loans = random.choice([0, 1])

    approve_chance = 0
    if credit_score > 700:
        approve_chance += 0.4
    elif credit_score > 600:
        approve_chance += 0.2
    if income > 30000:
        approve_chance += 0.2
    if employment_years > 2:
        approve_chance += 0.1
    if loan_amount < income * 0.4:
        approve_chance += 0.2
    if has_other_loans:
        approve_chance -= 0.2

    approved = int(random.random() < approve_chance)

    return {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "has_other_loans": has_other_loans,
        "approved": approved
    }


data = [generate_applicant() for _ in range(10000)]
df = pd.DataFrame(data)


os.makedirs("data", exist_ok=True)


df.to_csv("data/credit_data.csv", index=False)
print(" Збережено у 'data/credit_data.csv'")
