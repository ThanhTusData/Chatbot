from typing import Dict, List
import numpy as np
from nlp.processor import NLPProcessor
from nlp.tokenizer import CustomTokenizer
from classification.intent_classifier import IntentClassifier

class IntentPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.processor = NLPProcessor()
        self.tokenizer = CustomTokenizer.load(tokenizer_path)
        self.classifier = IntentClassifier(model_path)
    
    def predict(self, text: str, top_k: int = 3) -> List[Dict[str, float]]:
        # Preprocess
        processed_text = self.processor.preprocess_text(text)
        
        # Tokenize
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        
        # Predict
        results = self.classifier.predict(sequence, top_k=top_k)
        return results[0]
    
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[List[Dict[str, float]]]:
        processed_texts = [self.processor.preprocess_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        return self.classifier.predict(sequences, top_k=top_k)