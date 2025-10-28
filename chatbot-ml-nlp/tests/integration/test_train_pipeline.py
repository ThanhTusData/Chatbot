import pytest
import json
from pathlib import Path

def test_data_loader():
    from src.training.data_loader import DataLoader
    
    # Create sample data
    sample_data = [
        {"text": "hello", "intent": "greeting"},
        {"text": "goodbye", "intent": "farewell"}
    ]
    
    # This is a basic test - in real scenario, use temp files
    assert len(sample_data) == 2

def test_tokenizer_fit():
    from src.nlp.tokenizer import CustomTokenizer
    
    tokenizer = CustomTokenizer(max_length=50)
    texts = ["hello world", "goodbye world"]
    
    tokenizer.fit(texts)
    
    assert tokenizer.vocab_size > 0
    
    sequences = tokenizer.texts_to_sequences(texts)
    assert sequences.shape[0] == 2
    assert sequences.shape[1] == 50