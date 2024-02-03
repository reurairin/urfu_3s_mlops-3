import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("preprocess_data")

with mlflow.start_run():
    data_frame = pd.read_csv("/home/reurairin/projects/urfu/urfu_3s_mlops-3/data/iris.csv", sep=',')

    data_frame['variety'] = LabelEncoder().fit_transform(data_frame['variety'])

    save_dir = "/home/reurairin/projects/urfu/urfu_3s_mlops-3/data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "iris_prepared.csv")

    data_frame.to_csv(file_path, index=False)

    mlflow.log_artifact(local_path="/home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/preprocess_data.py",
                        artifact_path="preprocess_data code")
    
    mlflow.end_run()
