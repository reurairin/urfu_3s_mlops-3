import os
import pickle
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    data = pd.read_csv('/home/reurairin/projects/urfu/urfu_3s_mlops-3/data/test.csv', sep=',')

    model_path = '/home/reurairin/projects/urfu/urfu_3s_mlops-3/models/model.pickle'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    try:
        score = model.score(data.iloc[:, :-1], data.iloc[:, -1])
        print('score=', score)

        mlflow.log_artifact(local_path='/home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/test_model.py',
                            artifact_path="test_model code")
        mlflow.log_metric("score", score)

    except Exception as e:
        print("Error during model testing:", e)

    mlflow.end_run()
