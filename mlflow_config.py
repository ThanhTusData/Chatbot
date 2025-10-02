# mlflow_config.py
import mlflow

def init_mlflow(tracking_uri: str = None, experiment_name: str = "default"):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return True
