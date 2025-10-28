from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import threading

from config import Config
from src.core.chatbot import IntelligentChatbot

def create_desktop_app():
    """Tạo desktop app với PyQt5"""
    
    try:
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
