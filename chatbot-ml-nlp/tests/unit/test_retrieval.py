import pytest
import numpy as np
from src.retrieval.vectorstore import VectorStore
from src.retrieval.embed import EmbeddingGenerator

def test_vectorstore_add_documents(sample_embeddings):
    vs = VectorStore(embedding_dim=384)
    documents = [{"content": f"doc {i}"} for i in range(5)]
    
    vs.add_documents(sample_embeddings, documents)
    
    assert len(vs.documents) == 5

def test_vectorstore_search(sample_embeddings):
    vs = VectorStore(embedding_dim=384)
    documents = [{"content": f"doc {i}"} for i in range(5)]
    vs.add_documents(sample_embeddings, documents)
    
    query_emb = np.random.rand(384).astype('float32')
    results = vs.search(query_emb, top_k=3)
    
    assert len(results) <= 3
    assert all('score' in r for r in results)

def test_embedding_generator():
    embedder = EmbeddingGenerator()
    text = "Hello world"
    embedding = embedder.encode_single(text)
    
    assert embedding.shape[0] == embedder.embedding_dim
    assert isinstance(embedding, np.ndarray)

def test_compute_similarity():
    embedder = EmbeddingGenerator()
    similarity = embedder.compute_similarity("hello", "hi")
    
    assert 0 <= similarity <= 1
    assert isinstance(similarity, float)