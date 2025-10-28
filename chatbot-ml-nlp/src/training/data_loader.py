import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.label_encoder = LabelEncoder()
    
    def load_intent_data(self) -> Tuple[List[str], List[str]]:
        data = []
        labels = []
        
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            for item in dataset:
                if 'text' in item and 'intent' in item:
                    data.append(item['text'])
                    labels.append(item['intent'])
        
        elif self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if 'text' in item and 'intent' in item:
                        data.append(item['text'])
                        labels.append(item['intent'])
        
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            data = df['text'].tolist()
            labels = df['intent'].tolist()
        
        return data, labels
    
    def prepare_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        texts, labels = self.load_intent_data()
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_label_mapping(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}