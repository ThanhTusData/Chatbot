import numpy as np
from typing import List, Dict

from config import Config

class ResponseGenerator:
    """Tạo phản hồi thông minh với fallback mechanism"""
    
    def __init__(self, config: Config):
        self.config = config
        self.responses_db = {}
        self.fallback_responses = [
            "Tôi không hiểu rõ câu hỏi của bạn. Bạn có thể diễn đạt khác được không?",
            "Xin lỗi, tôi cần thêm thông tin để trả lời chính xác.",
            "Tôi đang học hỏi thêm về chủ đề này. Bạn có thể hỏi tôi về vấn đề khác không?"
        ]
        self.load_responses()
        
    def load_responses(self):
        """Tải cơ sở dữ liệu phản hồi"""
        sample_responses = {
            'greeting': [
                "Xin chào! Tôi có thể giúp gì cho bạn?",
                "Chào bạn! Tôi là chatbot hỗ trợ. Có điều gì tôi có thể giúp không?",
                "Xin chào! Rất vui được gặp bạn!"
            ],
            'product_inquiry': [
                "Tôi có thể giúp bạn tìm hiểu về sản phẩm. Bạn quan tâm đến loại sản phẩm nào?",
                "Chúng tôi có nhiều sản phẩm chất lượng. Bạn muốn biết thông tin gì cụ thể?"
            ],
            'technical_support': [
                "Tôi sẵn sàng hỗ trợ kỹ thuật. Bạn đang gặp vấn đề gì?",
                "Hãy mô tả chi tiết vấn đề để tôi có thể hỗ trợ tốt nhất."
            ],
            'goodbye': [
                "Tạm biệt! Chúc bạn một ngày tốt lành!",
                "Hẹn gặp lại! Nếu cần hỗ trợ thêm, đừng ngần ngại liên hệ.",
                "Bye bye! Cảm ơn bạn đã sử dụng dịch vụ!"
            ]
        }
        self.responses_db = sample_responses
    
    def generate_response(self, intent: str, confidence: float, 
                         user_message: str, context: List[Dict] = None) -> str:
        """Tạo phản hồi dựa trên intent và confidence"""
        
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            return self.handle_low_confidence(intent, user_message)
        
        if intent in self.responses_db:
            responses = self.responses_db[intent]
            return np.random.choice(responses)
        
        return self.get_fallback_response()
    
    def handle_low_confidence(self, predicted_intent: str, user_message: str) -> str:
        """Xử lý khi confidence thấp - gợi ý intent"""
        suggestions = self.suggest_intents(user_message)
        
        if suggestions:
            suggestion_text = ", ".join(suggestions[:3])
            return f"Tôi không chắc chắn về ý định của bạn. Bạn có muốn hỏi về: {suggestion_text}?"
        
        return self.get_fallback_response()
    
    def suggest_intents(self, user_message: str) -> List[str]:
        """Gợi ý intent dựa trên tương đồng văn bản"""
        # Implement intent suggestion logic
        available_intents = list(self.responses_db.keys())
        return available_intents[:3]  # Simplified
    
    def get_fallback_response(self) -> str:
        """Trả về phản hồi fallback"""
        return np.random.choice(self.fallback_responses)