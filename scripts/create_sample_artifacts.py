# scripts/create_sample_artifacts.py
import os
from src.models.intent_model import IntentModel
from src.retrieval.vectorstore import VectorStoreInMemory
import numpy as np
import json

# 1) create small intent model
texts = ["hello", "hi", "bye", "goodbye", "how to reset password", "i forgot password"]
labels = ["greeting", "greeting", "goodbye", "goodbye", "support", "support"]
m = IntentModel()
m.build()
m.fit(texts, labels)
out_dir = "models/intent/latest"
model_path, meta_path = m.save(out_dir)
print("Saved sample intent model to", out_dir)

# 2) create small index
ids = ["d1", "d2", "d3"]
docs = ["How to reset password", "Payment methods", "Account deletion process"]
# fake embeddings (small dim)
embs = np.random.rand(len(docs), 32).astype(float)
metas = [{"text": docs[i], "answer": ""} for i in range(len(docs))]
store = VectorStoreInMemory()
store.index(ids, docs, embs.tolist(), metadatas=metas)
idx_dir = "indexes/kb"
store.save(idx_dir)
print("Saved sample index to", idx_dir)
