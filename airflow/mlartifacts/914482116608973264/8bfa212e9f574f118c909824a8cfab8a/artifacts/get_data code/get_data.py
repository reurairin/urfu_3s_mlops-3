import requests
import mlflow
import os

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_data")

with mlflow.start_run():
    data = requests.get("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

    save_dir = "/home/reurairin/projects/urfu/urfu_3s_mlops-3/data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, "iris.csv")

    with open(file_path, "w") as f:
        f.write(data.text)

    mlflow.log_artifact(local_path="/home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/get_data.py",
                        artifact_path="get_data code")
    
    mlflow.end_run()