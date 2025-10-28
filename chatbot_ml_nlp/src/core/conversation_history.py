import json
from datetime import datetime
from typing import List, Dict

class ConversationHistory:
    """Quản lý lịch sử hội thoại"""
    
    def __init__(self):
        self.conversations = []
        self.session_id = None
        
    def add_message(self, user_message: str, bot_response: str, intent: str = None, confidence: float = None):
        message = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'confidence': confidence,
            'session_id': self.session_id
        }
        self.conversations.append(message)
        
    def get_recent_context(self, n_messages: int = 5) -> List[Dict]:
        return self.conversations[-n_messages:] if self.conversations else []
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)