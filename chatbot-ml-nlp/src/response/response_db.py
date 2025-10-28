import json
from pathlib import Path
from typing import Dict, List, Optional

class ResponseDatabase:
    def __init__(self, db_path: Optional[str] = None):
        self.responses: Dict[str, List[Dict]] = {}
        
        if db_path:
            self.load(db_path)
    
    def load(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.responses = data
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, indent=2, ensure_ascii=False)
    
    def add_response(self, intent: str, response: str, metadata: Optional[Dict] = None) -> None:
        if intent not in self.responses:
            self.responses[intent] = []
        
        response_obj = {
            'text': response,
            'metadata': metadata or {}
        }
        self.responses[intent].append(response_obj)
    
    def get_responses(self, intent: str) -> List[Dict]:
        return self.responses.get(intent, [])
    
    def remove_response(self, intent: str, index: int) -> bool:
        if intent in self.responses and 0 <= index < len(self.responses[intent]):
            del self.responses[intent][index]
            return True
        return False
    
    def get_all_intents(self) -> List[str]:
        return list(self.responses.keys())