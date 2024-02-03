import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("split_data")

with mlflow.start_run():
    data_frame = pd.read_csv('/home/reurairin/projects/urfu/urfu_3s_mlops-3/data/iris_prepared.csv', sep=',')

    X_train, X_test, y_train, y_test = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1], test_size=0.2, random_state=42)

    save_dir = '/home/reurairin/projects/urfu/urfu_3s_mlops-3/data'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pd.concat((X_train, y_train), axis=1).to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    pd.concat((X_test, y_test), axis=1).to_csv(os.path.join(save_dir, 'test.csv'), index=False)

    mlflow.log_artifact(local_path="/home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/split_data.py",
                        artifact_path="split_data code")

    mlflow.end_run()
