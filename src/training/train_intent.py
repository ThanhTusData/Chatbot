# src/training/train_intent.py
import argparse
import json
import logging
import os
import shutil
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score

# mlflow integration
import mlflow

from src.data.prepare import load_raw, clean, train_test_split_df
from src.models.intent_model import IntentModel
from src.mlflow_config import init_mlflow  # optional helper you have

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def train_from_config(
    data_path: str,
    out_root: str = "models/intent",
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_uri: str = None,
    experiment_name: str = "intent-training",
):
    # init mlflow if provided
    if mlflow_uri or experiment_name:
        try:
            init_mlflow(tracking_uri=mlflow_uri, experiment_name=experiment_name)
            LOG.info(f"MLflow initialized (uri={mlflow_uri}, experiment={experiment_name})")
        except Exception as e:
            LOG.warning(f"Failed to init MLflow: {e}")

    # load
    df = load_raw(data_path)
    df = clean(df)
    train_df, test_df = train_test_split_df(df, test_size=test_size, random_state=random_state)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(out_root, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # start mlflow run (so metrics & artifacts are grouped)
    with mlflow.start_run(run_name=f"train-{timestamp}"):
        # log params
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model = IntentModel()
        model.fit(train_df["text"].tolist(), train_df["intent"].tolist())

        # evaluate simple metric
        preds = model.predict(test_df["text"].tolist())
        acc = accuracy_score(test_df["intent"].tolist(), preds)
        LOG.info(f"Test accuracy: {acc:.4f}")

        # save model using existing save() method
        # Ensure IntentModel.save(model_dir) writes files under model_dir
        model.save(model_dir)

        # write metrics locally
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(acc)}, f, ensure_ascii=False, indent=2)

        # Log metric and artifacts to mlflow
        try:
            mlflow.log_metric("accuracy", float(acc))
            # log model artifacts (directory)
            mlflow.log_artifacts(model_dir, artifact_path="intent_model")
            LOG.info("Logged metrics and model artifacts to MLflow")
        except Exception as e:
            LOG.warning(f"Failed to log to MLflow: {e}")

    # update latest (copy)
    latest_dir = os.path.join(out_root, "latest")
    try:
        if os.path.exists(latest_dir):
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                shutil.rmtree(latest_dir)
    except Exception:
        pass
    shutil.copytree(model_dir, latest_dir)

    LOG.info(f"Saved model to {model_dir} and updated latest -> {latest_dir}")
    return {"model_dir": model_dir, "accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_data.json")
    parser.add_argument("--out", default="models/intent")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (e.g. file:./mlruns or http://mlflow:5000)")
    parser.add_argument("--experiment", default="intent-training")
    args = parser.parse_args()
    train_from_config(
        args.data,
        out_root=args.out,
        test_size=args.test_size,
        random_state=args.seed,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
