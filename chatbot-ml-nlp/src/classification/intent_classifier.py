import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

class IntentClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[keras.Model] = None
        self.label_encoder: Dict[int, str] = {}
        self.reverse_label_encoder: Dict[str, int] = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        self.model = keras.models.load_model(model_path)
        
        metadata_path = Path(model_path).parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.label_encoder = {int(k): v for k, v in metadata['label_encoder'].items()}
                self.reverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
    
    def predict(self, sequences: np.ndarray, top_k: int = 3) -> List[Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(sequences)
        results = []
        
        for pred in predictions:
            top_indices = np.argsort(pred)[-top_k:][::-1]
            intent_probs = [
                {
                    'intent': self.label_encoder[idx],
                    'confidence': float(pred[idx])
                }
                for idx in top_indices
            ]
            results.append(intent_probs)
        
        return results
    
    def predict_single(self, sequence: np.ndarray) -> Dict[str, float]:
        results = self.predict(np.array([sequence]), top_k=1)
        return results[0][0]
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not loaded")
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'precision': float(precision_score(y_test_classes, y_pred_classes, average='weighted')),
            'recall': float(recall_score(y_test_classes, y_pred_classes, average='weighted')),
            'f1_score': float(f1_score(y_test_classes, y_pred_classes, average='weighted'))
        }