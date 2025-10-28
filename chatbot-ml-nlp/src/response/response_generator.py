from typing import Dict, List, Optional
import random
import json
from pathlib import Path

class ResponseGenerator:
    def __init__(self, response_db_path: Optional[str] = None):
        self.responses: Dict[str, List[str]] = {}
        self.default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have information about that. Can you ask something else?",
            "Let me help you with something else.",
        ]
        
        if response_db_path:
            self.load_responses(response_db_path)
    
    def load_responses(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            self.responses = json.load(f)
    
    def generate(self, intent: str, confidence: float, context: Optional[Dict] = None) -> str:
        if confidence < 0.5:
            return random.choice(self.default_responses)
        
        if intent in self.responses:
            templates = self.responses[intent]
            response = random.choice(templates)
            
            if context:
                response = response.format(**context)
            
            return response
        
        return random.choice(self.default_responses)
    
    def generate_with_retrieval(
        self,
        intent: str,
        confidence: float,
        retrieved_docs: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        if confidence > 0.7 and retrieved_docs:
            best_doc = retrieved_docs[0]
            if 'answer' in best_doc:
                return best_doc['answer']
            elif 'content' in best_doc:
                return f"Based on our knowledge base: {best_doc['content']}"
        
        return self.generate(intent, confidence, context)