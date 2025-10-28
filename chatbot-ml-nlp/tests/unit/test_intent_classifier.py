import pytest
import numpy as np
from src.classification.intent_classifier import IntentClassifier
from src.classification.model_builder import IntentClassifierBuilder

def test_model_builder():
    model = IntentClassifierBuilder.build_lstm_attention_model(
        vocab_size=1000,
        embedding_dim=128,
        max_length=100,
        num_classes=10
    )
    
    assert model is not None
    assert len(model.layers) > 0

def test_cnn_model_builder():
    model = IntentClassifierBuilder.build_cnn_model(
        vocab_size=1000,
        embedding_dim=128,
        max_length=100,
        num_classes=10
    )
    
    assert model is not None
    assert len(model.layers) > 0

@pytest.fixture
def sample_model(tmp_path):
    model = IntentClassifierBuilder.build_lstm_attention_model(
        vocab_size=100,
        embedding_dim=32,
        max_length=50,
        num_classes=5
    )
    
    model_path = tmp_path / "test_model.h5"
    model.save(str(model_path))
    
    return str(model_path)

def test_intent_classifier_predict(sample_model):
    classifier = IntentClassifier(sample_model)
    
    # Create dummy input
    sequences = np.random.randint(0, 100, (5, 50))
    
    results = classifier.predict(sequences, top_k=3)
    
    assert len(results) == 5
    assert len(results[0]) == 3
    assert 'intent' in results[0][0]
    assert 'confidence' in results[0][0]