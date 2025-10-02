# scripts/build_kb_index.py
import argparse
import json
import os
from src.retrieval.embed import Embedder
from src.retrieval.vectorstore import VectorStoreInMemory
from datetime import datetime

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(input_path: str, out_dir: str, model_name: str = None):
    docs = []
    ids = []
    metas = []
    for item in load_jsonl(input_path):
        _id = item.get("id") or item.get("question")[:64]
        ids.append(str(_id))
        question = item.get("question", "")
        answer = item.get("answer", "")
        text = (question + " " + answer).strip()
        docs.append(text)
        metas.append({
            "question": question,
            "answer": answer,
            "source": item.get("source", "kb")
        })

    embedder = Embedder(model_name=model_name)
    embeddings = embedder.embed(docs)
    store = VectorStoreInMemory()
    store.index(ids, docs, embeddings, metadatas=metas)
    store.save(out_dir)

    meta = {
        "num_docs": len(ids),
        "embed_model": model_name or embedder.model_name,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kb_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Index built at {out_dir} ({len(ids)} docs)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="data/kb/faq.jsonl")
    parser.add_argument("--out", required=True, help="output index dir, e.g. indexes/kb")
    parser.add_argument("--model", required=False, help="embedding model name")
    args = parser.parse_args()
    main(args.input, args.out, args.model)
