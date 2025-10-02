import os
import json

from config import Config
from src.core.chatbot import IntelligentChatbot
from src.web.flask_app import ChatbotWebApp, SAMPLE_TRAINING_DATA

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