# src/retrieval/serve_helpers.py
from src.retrieval.embed import Embedder
from src.retrieval.vectorstore import VectorStoreInMemory
import os

_store: VectorStoreInMemory = None
_embedder: Embedder = None

def init(index_path: str, model_name: str = None):
    global _store, _embedder
    _embedder = Embedder(model_name=model_name)
    _store = VectorStoreInMemory()
    _store.load(index_path)

def retrieve_for_api(query: str, top_k: int = 5):
    if _embedder is None or _store is None:
        raise RuntimeError("Retrieval not initialized. Call init(index_path) first.")
    q_emb = _embedder.embed([query])[0]
    results = _store.search(q_emb, top_k=top_k)
    # optionally add the text/answer into response if stored in metadata
    out = []
    for r in results:
        meta = r.get("metadata", {})
        item = {
            "id": r["id"],
            "score": r["score"],
            "text": meta.get("text") or meta.get("question"),
            "answer": meta.get("answer")
        }
        out.append(item)
    return out
