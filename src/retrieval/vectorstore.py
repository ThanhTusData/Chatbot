# src/retrieval/vectorstore.py
from typing import List, Dict, Any
import numpy as np
import json
import os

class VectorStoreInMemory:
    """
    Simple in-memory vectorstore with brute-force cosine similarity search.
    Save/load as numpy + json files.
    """
    def __init__(self):
        self.ids: List[str] = []
        self.embs: np.ndarray = None  # shape (N, D)
        self.metadatas: Dict[str, Dict] = {}

    def index(self, ids: List[str], docs: List[str], embeddings: List[List[float]], metadatas: List[Dict]=None):
        embs = np.array(embeddings, dtype=float)
        if self.embs is None:
            self.embs = embs
        else:
            self.embs = np.vstack([self.embs, embs])
        self.ids.extend(ids)
        if metadatas:
            for i, _id in enumerate(ids):
                self.metadatas[_id] = metadatas[i]
        else:
            for i, _id in enumerate(ids):
                self.metadatas[_id] = {"text": docs[i]}

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embs is None or len(self.ids) == 0:
            return []
        q = np.array(query_embedding, dtype=float)
        # cosine similarity
        emb_norms = np.linalg.norm(self.embs, axis=1)
        q_norm = np.linalg.norm(q) + 1e-12
        sims = (self.embs @ q) / (emb_norms * q_norm + 1e-12)
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            _id = self.ids[i]
            meta = self.metadatas.get(_id, {})
            results.append({
                "id": _id,
                "score": float(sims[i]),
                "metadata": meta
            })
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.embs is not None:
            np.save(os.path.join(path, "embs.npy"), self.embs)
        with open(os.path.join(path, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "metadatas.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        embs_path = os.path.join(path, "embs.npy")
        ids_path = os.path.join(path, "ids.json")
        metas_path = os.path.join(path, "metadatas.json")
        if os.path.exists(embs_path):
            self.embs = np.load(embs_path)
        else:
            self.embs = None
        if os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                self.ids = json.load(f)
        else:
            self.ids = []
        if os.path.exists(metas_path):
            with open(metas_path, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)
        else:
            self.metadatas = {}
