from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt

args = {
        "owner": "admin",
        "start_date": dt.datetime(2023, 2, 1),
        "retries": 1,
        "retry_delays": dt.timedelta(minutes=1),
        "depends_on_past": False
        }
with DAG(dag_id='iris_score', default_args=args, schedule_interval=None, tags=['iris']) as dag:
    get_data = BashOperator(task_id='get_data',
            bash_command='python3 /home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/get_data.py',
            dag=dag)
    preprocess_data = BashOperator(task_id='prepare_data',
            bash_command='python3 /home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/preprocess_data.py',
            dag=dag)
    split_data = BashOperator(task_id='split_data',
            bash_command='python3 /home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/split_data.py',
            dag=dag)
    train_model = BashOperator(task_id='train_model',
            bash_command='python3 /home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/train_model.py',
            dag=dag)
    test_model = BashOperator(task_id='test_model',
            bash_command='python3 /home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/test_model.py',
            dag=dag)

    get_data >> preprocess_data >> split_data >> train_model >> test_model