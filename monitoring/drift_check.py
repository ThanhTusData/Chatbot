# monitoring/drift_check.py
"""
Simple drift check script:
- compares distribution of intents between a baseline dataset (training) and a current dataset (production sample)
- computes JS divergence and writes a small report (json + optional html)
- logs metric to MLflow if MLflow URI provided

Usage:
python monitoring/drift_check.py --baseline data/training_data.json --current data/production_sample.json --mlflow-uri file:./mlruns
"""
import argparse
import json
import os
from collections import Counter
from math import log
from typing import Dict

import pandas as pd
import mlflow

from src.models.intent_model import IntentModel


def load_labels_from_json(path: str, label_field: str = "intent"):
    df = pd.read_json(path, orient="records", lines=False)
    return df[label_field].astype(str).tolist()


def distribution(labels):
    c = Counter(labels)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}


def js_divergence(p: Dict[str, float], q: Dict[str, float]):
    # convert to same keys
    keys = set(p.keys()) | set(q.keys())
    import math

    def kl(a, b):
        s = 0.0
        for k in keys:
            pa = a.get(k, 0.0)
            pb = b.get(k, 0.0)
            if pa == 0:
                continue
            s += pa * math.log(pa / (pb if pb > 0 else 1e-12))
        return s

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline (training) json")
    parser.add_argument("--current", required=True, help="Path to current/production sample json")
    parser.add_argument("--model-dir", default="models/intent/latest", help="Path to intent model")
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLflow URI")
    parser.add_argument("--out", default="monitoring/reports", help="Output report dir")
    args = parser.parse_args()

    if args.mlflow_uri:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
        except Exception as e:
            print("Warning: could not set mlflow uri:", e)

    os.makedirs(args.out, exist_ok=True)

    baseline_labels = load_labels_from_json(args.baseline)
    current_labels = load_labels_from_json(args.current)

    p = distribution(baseline_labels)
    q = distribution(current_labels)

    js = js_divergence(p, q)
    report = {
        "baseline_counts": dict(Counter(baseline_labels)),
        "current_counts": dict(Counter(current_labels)),
        "js_divergence": js,
    }
    report_path = os.path.join(args.out, "drift_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # write a minimal html
    html_path = os.path.join(args.out, "drift_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body>")
        f.write(f"<h2>Drift report</h2>")
        f.write(f"<p>JS divergence: {js:.6f}</p>")
        f.write("<h3>Baseline counts</h3><pre>")
        f.write(json.dumps(report["baseline_counts"], indent=2))
        f.write("</pre><h3>Current counts</h3><pre>")
        f.write(json.dumps(report["current_counts"], indent=2))
        f.write("</pre></body></html>")

    print(f"Saved report to {report_path} and {html_path}")

    # log to mlflow if configured
    try:
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metric("js_divergence", float(js))
            mlflow.log_artifact(report_path, artifact_path="drift")
            mlflow.log_artifact(html_path, artifact_path="drift")
            print("Logged drift metrics to MLflow")
    except Exception as e:
        print("MLflow logging skipped/failed:", e)


if __name__ == "__main__":
    main()
