import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_texts():
    return [
        "Hello, how are you?",
        "What's the weather like today?",
        "Tell me about your product",
        "I need technical support",
        "Goodbye!"
    ]

@pytest.fixture
def sample_intents():
    return ["greeting", "weather", "product_inquiry", "technical_support", "goodbye"]

@pytest.fixture
def sample_embeddings():
    return np.random.rand(5, 384).astype('float32')

@pytest.fixture
def temp_model_dir(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir