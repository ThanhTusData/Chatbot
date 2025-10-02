# src/eval/evaluate_retrieval.py
import json
import os
from src.retrieval.serve_helpers import init, retrieve_for_api

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def evaluate(index_path: str, queries_path: str, top_k: int = 5, model_name: str = None):
    init(index_path, model_name=model_name)
    total = 0
    top1 = 0
    topk = 0
    mrr_total = 0.0
    for q in load_queries(queries_path):
        total += 1
        query_text = q["query"]
        relevant = set(q.get("relevant_ids", []))
        results = retrieve_for_api(query_text, top_k=top_k)
        found_ids = [r["id"] for r in results]
        # top1
        if found_ids and found_ids[0] in relevant:
            top1 += 1
        # topk
        if any(_id in relevant for _id in found_ids):
            topk += 1
        # MRR
        rr = 0.0
        for rank, _id in enumerate(found_ids, start=1):
            if _id in relevant:
                rr = 1.0 / rank
                break
        mrr_total += rr
    return {
        "queries": total,
        "top1": top1 / total if total else 0.0,
        "topk": topk / total if total else 0.0,
        "mrr": mrr_total / total if total else 0.0
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    res = evaluate(args.index, args.queries, top_k=args.top_k, model_name=args.model)
    print(json.dumps(res, indent=2))
