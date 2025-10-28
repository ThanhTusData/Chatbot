from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from typing import List, Optional
import numpy as np

class CustomTokenizer:
    def __init__(self, num_words: int = 10000, max_length: int = 100):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.max_length = max_length
        self.num_words = num_words
    
    def fit(self, texts: List[str]) -> None:
        self.tokenizer.fit_on_texts(texts)
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        return self.tokenizer.sequences_to_texts([sequence])[0]
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'max_length': self.max_length,
                'num_words': self.num_words
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'CustomTokenizer':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(num_words=data['num_words'], max_length=data['max_length'])
        instance.tokenizer = data['tokenizer']
        return instance
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.word_index) + 1