import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict

from tensorflow.keras.models import load_model

from config import Config
from nlp.processor import NLPProcessor
from classification.intent_classifier import IntentClassifier
from response.response_generator import ResponseGenerator
from voice.voice_processor import VoiceProcessor
from core.conversation_history import ConversationHistory

class IntelligentChatbot:
    """Chatbot chính với tất cả các tính năng"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp_processor = NLPProcessor(config)
        self.intent_classifier = IntentClassifier(config)
        self.response_generator = ResponseGenerator(config)
        self.voice_processor = VoiceProcessor()
        self.conversation_history = ConversationHistory()
        self.current_language = 'en'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def process_message(self, user_message: str, use_voice: bool = False) -> Dict:
        """Xử lý tin nhắn từ user"""
        
        # Ghi log
        self.logger.info(f"Processing message: {user_message}")
        
        # Tiền xử lý văn bản
        processed = self.nlp_processor.preprocess_text(user_message)
        
        # Phân loại intent
        intent, confidence = self.intent_classifier.predict_intent(processed['processed_text'])
        
        # Lấy context từ lịch sử
        context = self.conversation_history.get_recent_context()
        
        # Tạo phản hồi
        response = self.response_generator.generate_response(
            intent, confidence, user_message, context
        )
        
        # Lưu vào lịch sử
        self.conversation_history.add_message(
            user_message, response, intent, confidence
        )
        
        # Text-to-speech nếu cần
        if use_voice:
            self.voice_processor.text_to_speech(response)
        
        return {
            'user_message': user_message,
            'bot_response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': processed['entities'],
            'timestamp': datetime.now().isoformat()
        }
    
    def train_model(self, training_data_path: str):
        """Huấn luyện mô hình"""
        # Load training data
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Train intent classifier
        self.intent_classifier.train(training_data)
        
        # Save model
        self.save_model()
    
    def save_model(self):
        """Lưu mô hình"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        # Save TensorFlow model
        if self.intent_classifier.model:
            self.intent_classifier.model.save(
                os.path.join(self.config.MODEL_DIR, 'intent_model.h5')
            )
        
        # Save tokenizer và label encoder
        with open(os.path.join(self.config.MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.intent_classifier.tokenizer, f)
            
        with open(os.path.join(self.config.MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.intent_classifier.label_encoder, f)
    
    def load_model(self):
        """Tải mô hình đã huấn luyện"""
        try:
            # Load TensorFlow model
            model_path = os.path.join(self.config.MODEL_DIR, 'intent_model.h5')
            if os.path.exists(model_path):
                self.intent_classifier.model = load_model(model_path)
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.config.MODEL_DIR, 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.intent_classifier.tokenizer = pickle.load(f)
            
            # Load label encoder
            encoder_path = os.path.join(self.config.MODEL_DIR, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.intent_classifier.label_encoder = pickle.load(f)
                    
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")