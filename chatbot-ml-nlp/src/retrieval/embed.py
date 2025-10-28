from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))