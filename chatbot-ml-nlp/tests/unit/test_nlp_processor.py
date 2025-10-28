import pytest
from src.nlp.processor import NLPProcessor

def test_preprocess_text():
    processor = NLPProcessor()
    text = "Hello! How are you? Visit http://example.com"
    processed = processor.preprocess_text(text)
    
    assert "http" not in processed
    assert processed.islower()
    assert processed.strip() == processed

def test_tokenize(sample_texts):
    processor = NLPProcessor()
    tokens = processor.tokenize(sample_texts[0])
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)

def test_extract_entities():
    processor = NLPProcessor()
    text = "Apple Inc. is located in Cupertino, California"
    entities = processor.extract_entities(text)
    
    assert isinstance(entities, list)
    assert len(entities) > 0

def test_lemmatize():
    processor = NLPProcessor()
    text = "I am running and jumping"
    lemmas = processor.lemmatize(text)
    
    assert "run" in lemmas or "running" in lemmas
    assert isinstance(lemmas, list)