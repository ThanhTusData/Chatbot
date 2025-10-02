# tests/test_data_schema.py
import json
import os

def test_training_data_exists_and_schema():
    path = os.path.join("data", "training_data.json")
    assert os.path.exists(path), f"{path} not found. Create it with sample training data."

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "training_data.json should be a list"
    assert len(data) > 0, "training_data.json must have at least one sample"

    for item in data:
        assert isinstance(item, dict)
        assert "text" in item and "intent" in item
        assert isinstance(item["text"], str)
        assert isinstance(item["intent"], str)
