# tests/test_retrieval.py
import os
from src.retrieval.embed import Embedder
from src.retrieval.vectorstore import VectorStoreInMemory
from src.retrieval.serve_helpers import init, retrieve_for_api

def test_embed_and_vectorstore(tmp_path):
    docs = ["hello world", "forgot my password", "payment methods accept credit card"]
    ids = ["d1","d2","d3"]
    emb = Embedder()
    embs = emb.embed(docs)
    store = VectorStoreInMemory()
    metas = [{"text": docs[i], "answer": ""} for i in range(len(docs))]
    store.index(ids, docs, embs, metadatas=metas)
    # save/load roundtrip
    out = tmp_path / "idx"
    store.save(str(out))
    store2 = VectorStoreInMemory()
    store2.load(str(out))
    # search
    q_emb = emb.embed(["I forgot my password"])[0]
    res = store2.search(q_emb, top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "d2"
