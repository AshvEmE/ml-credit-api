FROM apache/airflow:2.7.2

COPY requirements.txt /requirements.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements.txt
