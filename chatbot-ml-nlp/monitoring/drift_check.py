import numpy as np
from typing import Dict, List
from scipy import stats
import json
from datetime import datetime

class DriftDetector:
    def __init__(self, baseline_data: List[Dict]):
        self.baseline_data = baseline_data
        self.baseline_stats = self._compute_stats(baseline_data)
    
    def _compute_stats(self, data: List[Dict]) -> Dict:
        if not data:
            return {}
        
        confidences = [d.get('confidence', 0) for d in data]
        intents = [d.get('intent', '') for d in data]
        
        return {
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'intent_distribution': self._get_distribution(intents)
        }
    
    def _get_distribution(self, items: List) -> Dict[str, float]:
        total = len(items)
        if total == 0:
            return {}
        
        dist = {}
        for item in items:
            dist[item] = dist.get(item, 0) + 1
        
        return {k: v/total for k, v in dist.items()}
    
    def detect_drift(self, new_data: List[Dict], threshold: float = 0.05) -> Dict:
        new_stats = self._compute_stats(new_data)
        
        baseline_conf = [d.get('confidence', 0) for d in self.baseline_data]
        new_conf = [d.get('confidence', 0) for d in new_data]
        
        ks_statistic, p_value = stats.ks_2samp(baseline_conf, new_conf)
        
        drift_detected = p_value < threshold
        
        return {
            'drift_detected': drift_detected,
            'p_value': float(p_value),
            'ks_statistic': float(ks_statistic),
            'baseline_stats': self.baseline_stats,
            'current_stats': new_stats,
            'timestamp': datetime.now().isoformat()
        }