from flask import Flask, request, jsonify, render_template
from src.core.chatbot import IntelligentChatbot

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