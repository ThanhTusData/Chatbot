# tests/test_train_pipeline.py
import os
import shutil
from src.training.train_intent import train_from_config

def test_train_pipeline(tmp_path):
    # use small sample from repo -> data/training_data.json
    data_path = os.path.join("data", "training_data.json")
    assert os.path.exists(data_path), "data/training_data.json missing for test"

    out_root = str(tmp_path / "models" / "intent")
    result = train_from_config(data_path, out_root=out_root, test_size=0.25, random_state=1)
    model_dir = result.get("model_dir")
    assert model_dir is not None
    assert os.path.exists(model_dir)
    # latest copy should exist
    latest = os.path.join(out_root, "latest")
    assert os.path.exists(latest)
    # metrics file exists
    metrics = os.path.join(model_dir, "metrics.json")
    assert os.path.exists(metrics)
    # accuracy is in result and between 0 and 1
    acc = result.get("accuracy")
    assert acc is not None
    assert 0.0 <= acc <= 1.0
