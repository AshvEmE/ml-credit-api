# Базовий образ
FROM python:3.10

# Робоча папка всередині контейнера
WORKDIR /app

# Копіюємо весь код
COPY . .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт
EXPOSE 8000

# Запускаємо FastAPI сервер
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
