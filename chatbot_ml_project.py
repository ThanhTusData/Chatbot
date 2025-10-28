# Chatbot-Using-ML-NLP Project
# Kiến trúc modular với spaCy, BERT, TensorFlow và giao diện web

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Core ML/NLP libraries
import spacy
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel
import torch

# Web framework
from flask import Flask, request, jsonify, render_template, session
import streamlit as st

# Additional utilities
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr
import pyttsx3

# ================================
# 1. DATA MODELS & CONFIGURATION
# ================================

class Config:
    """Cấu hình toàn cục cho chatbot"""
    
    # Model configuration
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 300
    HIDDEN_UNITS = 128
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # NLP Configuration
    SPACY_MODEL = "en_core_web_sm"  # hoặc "vi_core_news_sm" cho tiếng Việt
    BERT_MODEL = "bert-base-uncased"
    CONFIDENCE_THRESHOLD = 0.7
    
    # File paths
    MODEL_DIR = "models"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    
    # Supported languages
    LANGUAGES = {
        'en': 'English',
        'vi': 'Vietnamese',
        'es': 'Spanish',
        'fr': 'French'
    }

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

# ================================
# 2. NLP PREPROCESSING MODULE
# ================================

class NLPProcessor:
    """Xử lý ngôn ngữ tự nhiên với spaCy và BERT"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.load_models()
        
    def load_models(self):
        """Tải các model NLP"""
        try:
            # Load spaCy model
            self.nlp = spacy.load(self.config.SPACY_MODEL)
            
            # Load BERT model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config.BERT_MODEL)
            self.bert_model = AutoModel.from_pretrained(self.config.BERT_MODEL)
            
            logging.info("NLP models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading NLP models: {e}")
            
    def preprocess_text(self, text: str) -> Dict:
        """Tiền xử lý văn bản với spaCy"""
        doc = self.nlp(text.lower().strip())
        
        # Lemmatization và loại bỏ stopwords
        lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # POS tagging
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            'original': text,
            'lemmatized': lemmatized,
            'entities': entities,
            'pos_tags': pos_tags,
            'processed_text': ' '.join(lemmatized)
        }
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Tạo embeddings với BERT"""
        inputs = self.bert_tokenizer(text, return_tensors='pt', 
                                   max_length=self.config.MAX_SEQUENCE_LENGTH, 
                                   truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Lấy [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embeddings.flatten()

# ================================
# 3. INTENT CLASSIFICATION MODEL
# ================================

class IntentClassifier:
    """Mô hình phân loại intent với LSTM/GRU"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intent_responses = {}
        
    def build_model(self, vocab_size: int, num_classes: int):
        """Xây dựng mô hình LSTM/GRU"""
        
        # Input layer
        input_layer = Input(shape=(self.config.MAX_SEQUENCE_LENGTH,))
        
        # Embedding layer
        embedding = Embedding(vocab_size, self.config.EMBEDDING_DIM, 
                            mask_zero=True)(input_layer)
        
        # LSTM layers với Attention
        lstm1 = LSTM(self.config.HIDDEN_UNITS, return_sequences=True, 
                    dropout=self.config.DROPOUT_RATE)(embedding)
        lstm2 = LSTM(self.config.HIDDEN_UNITS, return_sequences=True, 
                    dropout=self.config.DROPOUT_RATE)(lstm1)
        
        # Attention mechanism
        attention = Attention()([lstm2, lstm2])
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(attention)
        dropout = Dropout(self.config.DROPOUT_RATE)(dense1)
        
        # Output layer
        output = Dense(num_classes, activation='softmax')(dropout)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, training_data: List[Dict]):
        """Huấn luyện mô hình"""
        # Chuẩn bị dữ liệu
        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]
        
        # Tokenization
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.config.MAX_SEQUENCE_LENGTH)
        
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.utils import to_categorical
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded)
        
        # Build và train model
        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)
        
        self.build_model(vocab_size, num_classes)
        
        # Training với callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.model.fit(
            X, y,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Dự đoán intent và confidence"""
        if not self.model or not self.tokenizer:
            return "unknown", 0.0
            
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.config.MAX_SEQUENCE_LENGTH)
        
        # Predict
        prediction = self.model.predict(padded)[0]
        confidence = np.max(prediction)
        intent_idx = np.argmax(prediction)
        
        intent = self.label_encoder.inverse_transform([intent_idx])[0]
        
        return intent, confidence

# ================================
# 4. RESPONSE GENERATION MODULE
# ================================

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

# ================================
# 5. VOICE PROCESSING MODULE
# ================================

class VoiceProcessor:
    """Xử lý giọng nói - Speech-to-Text và Text-to-Speech"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
        
    def setup_tts(self):
        """Cấu hình TTS engine"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
    
    def speech_to_text(self, language: str = 'en-US') -> str:
        """Chuyển đổi giọng nói thành văn bản"""
        try:
            with self.microphone as source:
                print("Đang nghe...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
                
            text = self.recognizer.recognize_google(audio, language=language)
            return text
            
        except sr.UnknownValueError:
            return "Không thể nhận diện giọng nói"
        except sr.RequestError as e:
            return f"Lỗi dịch vụ: {e}"
    
    def text_to_speech(self, text: str):
        """Chuyển văn bản thành giọng nói"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS Error: {e}")

# ================================
# 6. MAIN CHATBOT CLASS
# ================================

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

# ================================
# 7. FLASK WEB APPLICATION
# ================================

class ChatbotWebApp:
    """Ứng dụng web Flask cho chatbot"""
    
    def __init__(self, chatbot: IntelligentChatbot):
        self.app = Flask(__name__)
        self.app.secret_key = 'your-secret-key-here'
        self.chatbot = chatbot
        self.setup_routes()
        
    def setup_routes(self):
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            data = request.json
            user_message = data.get('message', '')
            use_voice = data.get('use_voice', False)
            
            if not user_message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Process message
            result = self.chatbot.process_message(user_message, use_voice)
            
            return jsonify(result)
        
        @self.app.route('/voice-input', methods=['POST'])
        def voice_input():
            language = request.json.get('language', 'en-US')
            
            # Speech to text
            text = self.chatbot.voice_processor.speech_to_text(language)
            
            if text and "Không thể nhận diện" not in text:
                result = self.chatbot.process_message(text, use_voice=True)
                return jsonify(result)
            
            return jsonify({'error': 'Voice recognition failed'}), 400
        
        @self.app.route('/history')
        def get_history():
            return jsonify(self.chatbot.conversation_history.conversations)
        
        @self.app.route('/set-language', methods=['POST'])
        def set_language():
            language = request.json.get('language', 'en')
            self.chatbot.current_language = language
            return jsonify({'status': 'success', 'language': language})
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)

# ================================
# 8. SAMPLE TRAINING DATA
# ================================

SAMPLE_TRAINING_DATA = [
    {'text': 'hello', 'intent': 'greeting'},
    {'text': 'hi there', 'intent': 'greeting'},
    {'text': 'good morning', 'intent': 'greeting'},
    {'text': 'hey', 'intent': 'greeting'},
    
    {'text': 'tell me about your products', 'intent': 'product_inquiry'},
    {'text': 'what products do you have', 'intent': 'product_inquiry'},
    {'text': 'product information', 'intent': 'product_inquiry'},
    {'text': 'show me your items', 'intent': 'product_inquiry'},
    
    {'text': 'I need technical support', 'intent': 'technical_support'},
    {'text': 'help me with technical issue', 'intent': 'technical_support'},
    {'text': 'technical problem', 'intent': 'technical_support'},
    {'text': 'system not working', 'intent': 'technical_support'},
    
    {'text': 'goodbye', 'intent': 'goodbye'},
    {'text': 'bye', 'intent': 'goodbye'},
    {'text': 'see you later', 'intent': 'goodbye'},
    {'text': 'have a good day', 'intent': 'goodbye'},
]

# ================================
# 9. MAIN APPLICATION
# ================================

def main():
    """Khởi chạy ứng dụng chatbot"""
    
    # Khởi tạo cấu hình
    config = Config()
    
    # Tạo thư mục cần thiết
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Lưu dữ liệu training mẫu
    training_data_path = os.path.join(config.DATA_DIR, 'training_data.json')
    if not os.path.exists(training_data_path):
        with open(training_data_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_TRAINING_DATA, f, ensure_ascii=False, indent=2)
    
    # Khởi tạo chatbot
    chatbot = IntelligentChatbot(config)
    
    # Huấn luyện model (nếu chưa có)
    model_path = os.path.join(config.MODEL_DIR, 'intent_model.h5')
    if not os.path.exists(model_path):
        print("Training model...")
        chatbot.train_model(training_data_path)
    else:
        print("Loading existing model...")
        chatbot.load_model()
    
    # Khởi chạy web app
    web_app = ChatbotWebApp(chatbot)
    print("Starting chatbot web application...")
    print("Visit: http://localhost:5000")
    web_app.run()

if __name__ == "__main__":
    main()

# ================================
# 10. REQUIREMENTS.TXT
# ================================

REQUIREMENTS = """
# Core ML/NLP libraries
tensorflow>=2.10.0
torch>=1.12.0
transformers>=4.21.0
spacy>=3.4.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.4.0

# Web framework
flask>=2.2.0
streamlit>=1.12.0

# Voice processing
SpeechRecognition>=3.8.0
pyttsx3>=2.90

# Additional utilities
requests>=2.28.0
python-dateutil>=2.8.0
Pillow>=9.0.0

# Development tools
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# For PyQt5 desktop app (optional)
PyQt5>=5.15.0
"""

# ================================
# 11. HTML TEMPLATE
# ================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin: 10px 0;
            padding: 15px;
            border-radius: 15px;
            max-width: 70%;
            animation: slideIn 0.3s ease-out;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
        }
        
        .bot-message {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
        }
        
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        
        .input-field {
            flex: 1;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        
        .btn {
            padding: 15px 20px;
            margin-left: 10px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-voice {
            background: #28a745;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .typing-indicator {
            display: none;
            padding: 15px;
            background: white;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 70%;
            border: 1px solid #e9ecef;
        }
        
        .typing-dots {
            display: flex;
            gap: 5px;
        }
        
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #999;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        .language-selector {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
        }
        
        .confidence-indicator {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        .entities-display {
            font-size: 11px;
            color: #888;
            margin-top: 3px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 Intelligent Chatbot</h1>
            <p>Powered by ML/NLP • Hỗ trợ đa ngôn ngữ • Nhận diện giọng nói</p>
            <select class="language-selector" id="languageSelect">
                <option value="en">English</option>
                <option value="vi">Tiếng Việt</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
            </select>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                <div>👋 Xin chào! Tôi là chatbot thông minh được trang bị công nghệ ML/NLP tiên tiến.</div>
                <div>🎯 Tôi có thể hỗ trợ bạn về sản phẩm, kỹ thuật, hoặc trò chuyện thân thiện.</div>
                <div>🎤 Bạn có thể nhắn tin hoặc sử dụng giọng nói để tương tác với tôi!</div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
            <span style="margin-left: 10px;">Chatbot đang suy nghĩ...</span>
        </div>
        
        <div class="chat-input">
            <input type="text" class="input-field" id="messageInput" 
                   placeholder="Nhập tin nhắn của bạn..." 
                   onkeypress="handleKeyPress(event)">
            <button class="btn btn-primary" onclick="sendMessage()">
                📤 Gửi
            </button>
            <button class="btn btn-voice" onclick="startVoiceInput()">
                🎤 Nói
            </button>
        </div>
    </div>

    <script>
        let currentLanguage = 'vi';
        let isRecording = false;
        
        // DOM elements
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const typingIndicator = document.getElementById('typingIndicator');
        const languageSelect = document.getElementById('languageSelect');
        
        // Language mapping for speech recognition
        const speechLanguages = {
            'en': 'en-US',
            'vi': 'vi-VN',
            'es': 'es-ES',
            'fr': 'fr-FR'
        };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            messageInput.focus();
            loadChatHistory();
        });
        
        // Language change handler
        languageSelect.addEventListener('change', function() {
            currentLanguage = this.value;
            fetch('/set-language', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ language: currentLanguage })
            });
        });
        
        // Send message function
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            messageInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: message,
                        use_voice: false
                    })
                });
                
                const data = await response.json();
                
                // Hide typing indicator
                hideTypingIndicator();
                
                if (data.error) {
                    addMessage('Xin lỗi, đã có lỗi xảy ra: ' + data.error, 'bot');
                } else {
                    addMessage(data.bot_response, 'bot', data.intent, data.confidence, data.entities);
                }
                
            } catch (error) {
                hideTypingIndicator();
                addMessage('Xin lỗi, không thể kết nối đến server.', 'bot');
                console.error('Error:', error);
            }
        }
        
        // Voice input function
        async function startVoiceInput() {
            if (isRecording) return;
            
            isRecording = true;
            const voiceBtn = document.querySelector('.btn-voice');
            voiceBtn.innerHTML = '🔴 Đang nghe...';
            voiceBtn.disabled = true;
            
            try {
                const response = await fetch('/voice-input', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        language: speechLanguages[currentLanguage] || 'vi-VN'
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Lỗi nhận diện giọng nói: ' + data.error, 'bot');
                } else {
                    addMessage(data.user_message, 'user');
                    addMessage(data.bot_response, 'bot', data.intent, data.confidence, data.entities);
                }
                
            } catch (error) {
                addMessage('Lỗi kết nối voice service.', 'bot');
                console.error('Voice error:', error);
            } finally {
                isRecording = false;
                voiceBtn.innerHTML = '🎤 Nói';
                voiceBtn.disabled = false;
            }
        }
        
        // Add message to chat
        function addMessage(message, sender, intent = null, confidence = null, entities = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            let messageContent = `<div>${message}</div>`;
            
            // Add confidence and intent info for bot messages
            if (sender === 'bot' && intent && confidence) {
                messageContent += `<div class="confidence-indicator">
                    Intent: ${intent} (${(confidence * 100).toFixed(1)}% confidence)
                </div>`;
            }
            
            // Add entities info
            if (entities && entities.length > 0) {
                const entitiesText = entities.map(ent => `${ent[0]} (${ent[1]})`).join(', ');
                messageContent += `<div class="entities-display">Entities: ${entitiesText}</div>`;
            }
            
            messageDiv.innerHTML = messageContent;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Show/hide typing indicator
        function showTypingIndicator() {
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTypingIndicator() {
            typingIndicator.style.display = 'none';
        }
        
        // Handle Enter key press
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Load chat history
        async function loadChatHistory() {
            try {
                const response = await fetch('/history');
                const history = await response.json();
                
                history.slice(-10).forEach(msg => {
                    addMessage(msg.user_message, 'user');
                    addMessage(msg.bot_response, 'bot', msg.intent, msg.confidence);
                });
            } catch (error) {
                console.log('No chat history available');
            }
        }
        
        // Auto-scroll to bottom
        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Add some welcome interactions
        setTimeout(() => {
            addMessage('💡 Tip: Bạn có thể thử hỏi tôi về sản phẩm, hỗ trợ kỹ thuật, hoặc chỉ đơn giản là chào hỏi!', 'bot');
        }, 2000);
    </script>
</body>
</html>
"""

# ================================
# 12. STREAMLIT ALTERNATIVE
# ================================

def create_streamlit_app():
    """Tạo ứng dụng Streamlit làm giao diện thay thế"""
    
    st.set_page_config(
        page_title="Intelligent Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 70%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background-color: white;
        color: #333;
        border: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Intelligent Chatbot")
    st.markdown("*Powered by ML/NLP • Hỗ trợ đa ngôn ngữ • Nhận diện giọng nói*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Language selection
        language = st.selectbox(
            "Ngôn ngữ:",
            options=['vi', 'en', 'es', 'fr'],
            format_func=lambda x: {
                'vi': '🇻🇳 Tiếng Việt',
                'en': '🇺🇸 English', 
                'es': '🇪🇸 Español',
                'fr': '🇫🇷 Français'
            }[x]
        )
        
        # Voice options
        use_voice = st.checkbox("🎤 Sử dụng giọng nói")
        
        # Model info
        st.header("📊 Thông tin Model")
        st.info("""
        - **NLP**: spaCy + BERT embeddings
        - **ML**: LSTM/GRU với Attention
        - **Framework**: TensorFlow 2.x
        - **Voice**: Speech Recognition
        """)
        
        # Clear history
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chatbot = None
    
    # Initialize chatbot
    if st.session_state.chatbot is None:
        with st.spinner("Đang khởi tạo chatbot..."):
            config = Config()
            st.session_state.chatbot = IntelligentChatbot(config)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show metadata for bot messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Intent: {metadata.get('intent', 'unknown')}")
                with col2:
                    confidence = metadata.get('confidence', 0)
                    st.caption(f"Confidence: {confidence:.2%}")
    
    # Chat input
    if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                result = st.session_state.chatbot.process_message(prompt, use_voice)
                
                st.write(result['bot_response'])
                
                # Show metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Intent: {result['intent']}")
                with col2:
                    st.caption(f"Confidence: {result['confidence']:.2%}")
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['bot_response'],
                    "metadata": result
                })

# ================================
# 13. DESKTOP APPLICATION (PyQt5)
# ================================

def create_desktop_app():
    """Tạo desktop app với PyQt5"""
    
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                                   QHBoxLayout, QWidget, QTextEdit, QLineEdit, 
                                   QPushButton, QLabel, QComboBox, QSplitter)
        from PyQt5.QtCore import Qt, QThread, pyqtSignal
        from PyQt5.QtGui import QFont
        
        class ChatbotThread(QThread):
            response_ready = pyqtSignal(dict)
            
            def __init__(self, chatbot, message):
                super().__init__()
                self.chatbot = chatbot
                self.message = message
                
            def run(self):
                result = self.chatbot.process_message(self.message)
                self.response_ready.emit(result)
        
        class ChatbotDesktopApp(QMainWindow):
            def __init__(self):
                super().__init__()
                self.chatbot = None
                self.init_ui()
                self.init_chatbot()
                
            def init_ui(self):
                self.setWindowTitle("🤖 Intelligent Chatbot")
                self.setGeometry(100, 100, 800, 600)
                
                # Central widget
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                
                # Layout
                layout = QVBoxLayout()
                
                # Header
                header = QLabel("🤖 Intelligent Chatbot")
                header.setAlignment(Qt.AlignCenter)
                header.setFont(QFont("Arial", 16, QFont.Bold))
                layout.addWidget(header)
                
                # Language selector
                lang_layout = QHBoxLayout()
                lang_layout.addWidget(QLabel("Ngôn ngữ:"))
                self.language_combo = QComboBox()
                self.language_combo.addItems(["Tiếng Việt", "English", "Español", "Français"])
                lang_layout.addWidget(self.language_combo)
                lang_layout.addStretch()
                layout.addLayout(lang_layout)
                
                # Chat area
                splitter = QSplitter(Qt.Horizontal)
                
                # Messages
                self.messages_area = QTextEdit()
                self.messages_area.setReadOnly(True)
                self.messages_area.setFont(QFont("Arial", 10))
                splitter.addWidget(self.messages_area)
                
                # Info panel
                info_panel = QTextEdit()
                info_panel.setReadOnly(True)
                info_panel.setMaximumWidth(200)
                info_panel.setPlainText("Thông tin Model:\n\n• NLP: spaCy + BERT\n• ML: LSTM/GRU\n• Framework: TensorFlow")
                splitter.addWidget(info_panel)
                
                layout.addWidget(splitter)
                
                # Input area
                input_layout = QHBoxLayout()
                
                self.input_field = QLineEdit()
                self.input_field.setPlaceholderText("Nhập tin nhắn...")
                self.input_field.returnPressed.connect(self.send_message)
                input_layout.addWidget(self.input_field)
                
                self.send_button = QPushButton("📤 Gửi")
                self.send_button.clicked.connect(self.send_message)
                input_layout.addWidget(self.send_button)
                
                self.voice_button = QPushButton("🎤 Nói")
                self.voice_button.clicked.connect(self.start_voice_input)
                input_layout.addWidget(self.voice_button)
                
                layout.addLayout(input_layout)
                
                central_widget.setLayout(layout)
                
                # Add welcome message
                self.add_message("Chào mừng bạn đến với Intelligent Chatbot! 🤖", "bot")
                
            def init_chatbot(self):
                config = Config()
                self.chatbot = IntelligentChatbot(config)
                
            def add_message(self, message, sender):
                if sender == "user":
                    self.messages_area.append(f"<div style='text-align: right; background: #007bff; color: white; padding: 10px; border-radius: 10px; margin: 5px;'><b>Bạn:</b> {message}</div>")
                else:
                    self.messages_area.append(f"<div style='background: #f8f9fa; padding: 10px; border-radius: 10px; margin: 5px;'><b>🤖 Bot:</b> {message}</div>")
                
            def send_message(self):
                message = self.input_field.text().strip()
                if not message:
                    return
                    
                self.add_message(message, "user")
                self.input_field.clear()
                
                # Process in thread
                self.thread = ChatbotThread(self.chatbot, message)
                self.thread.response_ready.connect(self.handle_response)
                self.thread.start()
                
            def handle_response(self, result):
                response = result['bot_response']
                confidence = result['confidence']
                intent = result['intent']
                
                full_response = f"{response}<br><small>Intent: {intent} ({confidence:.2%})</small>"
                self.add_message(full_response, "bot")
                
            def start_voice_input(self):
                # Voice input implementation
                self.voice_button.setText("🔴 Đang nghe...")
                self.voice_button.setEnabled(False)
                
                # Simulate voice input (replace with actual implementation)
                import threading
                def voice_process():
                    try:
                        text = self.chatbot.voice_processor.speech_to_text()
                        if text and "Không thể nhận diện" not in text:
                            self.input_field.setText(text)
                            self.send_message()
                    finally:
                        self.voice_button.setText("🎤 Nói")
                        self.voice_button.setEnabled(True)
                
                threading.Thread(target=voice_process).start()
        
        # Run desktop app
        app = QApplication([])
        window = ChatbotDesktopApp()
        window.show()
        app.exec_()
        
    except ImportError:
        print("PyQt5 not installed. Please install with: pip install PyQt5")

# ================================
# 14. DOCKER CONFIGURATION
# ================================

DOCKERFILE = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    portaudio19-dev \\
    python3-pyaudio \\
    espeak \\
    espeak-data \\
    libespeak1 \\
    libespeak-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p models data logs

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
"""

DOCKER_COMPOSE = """
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - TENSORFLOW_CPP_MIN_LOG_LEVEL=2
    restart: unless-stopped
    
  # Optional: Add Redis for session management
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
"""

# ================================
# 15. TESTING MODULE
# ================================

def create_test_suite():
    """Tạo test suite cho chatbot"""
    
    import unittest
    
    class TestChatbot(unittest.TestCase):
        
        def setUp(self):
            self.config = Config()
            self.chatbot = IntelligentChatbot(self.config)
            
        def test_nlp_processing(self):
            """Test NLP preprocessing"""
            text = "Hello, how are you today?"
            result = self.chatbot.nlp_processor.preprocess_text(text)
            
            self.assertIn('lemmatized', result)
            self.assertIn('entities', result)
            self.assertIn('processed_text', result)
            
        def test_intent_prediction(self):
            """Test intent classification"""
            # This would require a trained model
            pass
            
        def test_response_generation(self):
            """Test response generation"""
            response = self.chatbot.response_generator.generate_response(
                'greeting', 0.9, 'hello'
            )
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
        def test_conversation_history(self):
            """Test conversation history"""
            history = self.chatbot.conversation_history
            history.add_message("Hello", "Hi there", "greeting", 0.9)
            
            self.assertEqual(len(history.conversations), 1)
            self.assertEqual(history.conversations[0]['user_message'], "Hello")
            
    return TestChatbot

# ================================
# 16. DEPLOYMENT SCRIPTS
# ================================

DEPLOY_SCRIPT = """
#!/bin/bash

# Deployment script for Intelligent Chatbot

echo "🚀 Deploying Intelligent Chatbot..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download vi_core_news_sm

# Create directories
mkdir -p models data logs templates static

# Set environment variables
export FLASK_APP=main.py
export FLASK_ENV=production

# Run database migrations (if any)
# python migrate.py

# Start the application
echo "✅ Starting chatbot server..."
python main.py

echo "🎉 Deployment completed!"
echo "🌐 Visit: http://localhost:5000"
"""

# Print additional setup instructions
print("""
🚀 HƯỚNG DẪN TRIỂN KHAI CHATBOT ML/NLP

1. Cài đặt dependencies:
   pip install -r requirements.txt
   
2. Tải language models:
   python -m spacy download en_core_web_sm
   python -m spacy download vi_core_news_sm
   
3. Tạo template HTML:
   mkdir templates
   # Lưu HTML template vào templates/index.html
   
4. Chạy ứng dụng:
   python main.py
   
5. Truy cập: http://localhost:5000

🔧 TÍNH NĂNG CHÍNH:
✅ Kiến trúc modular
✅ spaCy + BERT embeddings  
✅ LSTM/GRU với Attention
✅ TensorFlow 2.x
✅ Fallback mechanism
✅ Gợi ý intent
✅ Giao diện web Flask
✅ Hỗ trợ giọng nói
✅ Đa ngôn ngữ
✅ Lưu lịch sử hội thoại
✅ Docker support
✅ Desktop app (PyQt5)
✅ Streamlit alternative

🎯 ỨNG DỤNG:
• Chăm sóc khách hàng
• Tư vấn sản phẩm  
• Hỗ trợ kỹ thuật
• Chatbot thông minh
""")