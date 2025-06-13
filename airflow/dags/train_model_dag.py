from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime



import model_training


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

def retrain_model():
    model_training.main()

with DAG(
    dag_id='retrain_credit_model',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['ml'],
) as dag:
    retrain = PythonOperator(
        task_id='train_model',
        python_callable=retrain_model,
    )
