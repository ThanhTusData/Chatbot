import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Dict] = []
        self.metadata: List[Dict] = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings of dim {self.embedding_dim}, got {embeddings.shape[1]}")
        
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['score'] = float(1 / (1 + dist))
                result['distance'] = float(dist)
                results.append(result)
        
        return results
    
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        with open(path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_documents': len(self.documents)
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        path = Path(directory)
        
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        instance = cls(embedding_dim=metadata['embedding_dim'])
        instance.index = faiss.read_index(str(path / "index.faiss"))
        
        with open(path / "documents.json", 'r', encoding='utf-8') as f:
            instance.documents = json.load(f)
        
        return instance