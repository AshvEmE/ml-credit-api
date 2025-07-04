
FROM python:3.10


WORKDIR /app


COPY . .


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000


RUN python model_training.py


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
