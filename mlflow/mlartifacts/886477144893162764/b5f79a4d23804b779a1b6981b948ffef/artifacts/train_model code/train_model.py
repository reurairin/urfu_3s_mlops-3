import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_model")

with mlflow.start_run():

    data = pd.read_csv('/home/reurairin/projects/urfu/urfu_3s_mlops-3/data/train.csv', sep=',')
    model = KNeighborsClassifier(n_neighbors=5)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model.fit(X, y)

    mlflow.sklearn.log_model(model, artifact_path="knn", registered_model_name="KNN Model")

    mlflow.log_artifact(local_path="/home/reurairin/projects/urfu/urfu_3s_mlops-3/scripts/train_model.py",
                        artifact_path="train_model code")

    model_dir = '/home/reurairin/projects/urfu/urfu_3s_mlops-3/models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'model.pickle'), 'wb') as f:
        pickle.dump(model, f)

    mlflow.end_run()
