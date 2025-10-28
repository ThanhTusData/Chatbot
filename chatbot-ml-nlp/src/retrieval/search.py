from typing import List, Dict
from retrieval.embed import EmbeddingGenerator
from retrieval.vectorstore import VectorStore

class SemanticSearch:
    def __init__(self, vectorstore_path: str, embedding_model: str):
        self.embedder = EmbeddingGenerator(embedding_model)
        self.vectorstore = VectorStore.load(vectorstore_path)
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        query_embedding = self.embedder.encode_single(query)
        results = self.vectorstore.search(query_embedding, top_k=top_k)
        
        filtered_results = [r for r in results if r['score'] >= threshold]
        return filtered_results
    
    def hybrid_search(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        results = self.search(query, top_k=top_k * 2)
        
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    if key in result and result[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results
        
        return results[:top_k]