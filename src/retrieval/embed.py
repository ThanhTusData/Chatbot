# src/retrieval/embed.py
from typing import List
import numpy as np
from config import Config
import os
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Abstraction embedder. Try to use sentence-transformers (all-MiniLM-L6-v2) if available,
    otherwise use a simple fallback hashing vector (dev only).
    """
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or getattr(Config, "EMBED_MODEL", "all-MiniLM-L6-v2")
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        # nếu chạy trên CI hoặc ép fallback thì không load sentence-transformers
        if os.getenv("CI_FALLBACK_EMBEDDER", "false").lower() in ("1","true","yes"):
            self._model = None
            return
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        if self._model:
            arr = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            # convert to python lists for easier json/numpy handling
            return arr.tolist()
        # fallback: deterministic simple vector
        return [self._fallback_vector(t) for t in texts]

    def _fallback_vector(self, text: str, dim: int = 128):
        v = np.zeros(dim, dtype=float)
        for i, ch in enumerate(text[:dim*4]):
            v[i % dim] += ord(ch)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.tolist()
