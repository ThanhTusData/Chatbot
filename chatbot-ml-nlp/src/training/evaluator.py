import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List
import json

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str] = None
    ) -> Dict:
        # Convert to class indices if one-hot encoded
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        self.metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Per-class metrics
        if label_names:
            self.metrics['per_class'] = {}
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for idx, label in enumerate(label_names):
                if idx < len(precision_per_class):
                    self.metrics['per_class'][label] = {
                        'precision': float(precision_per_class[idx]),
                        'recall': float(recall_per_class[idx]),
                        'f1_score': float(f1_per_class[idx])
                    }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        return self.metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str] = None
    ) -> str:
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        return classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0
        )
    
    def save_metrics(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)