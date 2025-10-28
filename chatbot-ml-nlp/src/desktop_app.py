import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QStatusBar, QMenuBar,
    QAction, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
import requests
from datetime import datetime
from config.config import config

class ChatThread(QThread):
    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, message, api_url):
        super().__init__()
        self.message = message
        self.api_url = api_url
    
    def run(self):
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={"message": self.message},
                timeout=10
            )
            
            if response.status_code == 200:
                self.response_received.emit(response.json())
            else:
                self.error_occurred.emit(f"API Error: {response.status_code}")
        except Exception as e:
            self.error_occurred.emit(str(e))

class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api_url = f"http://{config.API_HOST}:{config.API_PORT}"
        self.chat_history = []
        self.init_ui()
        self.check_api_status()
    
    def init_ui(self):
        self.setWindowTitle("ML/NLP Chatbot")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 11))
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        file_menu.addAction(clear_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Initial message
        self.add_bot_message("Hello! How can I help you today?")
    
    def check_api_status(self):
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.status_bar.showMessage("Connected to API")
            else:
                self.status_bar.showMessage("API Status: Unknown")
        except:
            self.status_bar.showMessage("Warning: Cannot connect to API")
    
    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return
        
        self.add_user_message(message)
        self.message_input.clear()
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("Sending...")
        
        # Create thread for API call
        self.chat_thread = ChatThread(message, self.api_url)
        self.chat_thread.response_received.connect(self.handle_response)
        self.chat_thread.error_occurred.connect(self.handle_error)
        self.chat_thread.start()
    
    def add_user_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.append(f"<p style='color: #2196F3;'><b>You ({timestamp}):</b></p>")
        self.chat_display.append(f"<p>{message}</p>")
        self.chat_display.append("<br>")
        self.chat_history.append({"role": "user", "message": message, "timestamp": timestamp})
    
    def add_bot_message(self, message, intent=None, confidence=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.append(f"<p style='color: #4CAF50;'><b>Bot ({timestamp}):</b></p>")
        self.chat_display.append(f"<p>{message}</p>")
        
        if intent and confidence:
            self.chat_display.append(
                f"<p style='color: #757575; font-size: 9pt;'>"
                f"Intent: {intent} (Confidence: {confidence:.2%})</p>"
            )
        
        self.chat_display.append("<br>")
        self.chat_history.append({
            "role": "bot",
            "message": message,
            "timestamp": timestamp,
            "intent": intent,
            "confidence": confidence
        })
        
        # Scroll to bottom
        self.chat_display.moveCursor(QTextCursor.End)
    
    def handle_response(self, data):
        self.add_bot_message(
            data['response'],
            intent=data['intent']['intent'],
            confidence=data['intent']['confidence']
        )
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Ready")
    
    def handle_error(self, error_message):
        self.add_bot_message(f"Error: {error_message}")
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Error occurred")
    
    def clear_chat(self):
        reply = QMessageBox.question(
            self,
            "Clear Chat",
            "Are you sure you want to clear the chat history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            self.chat_history.clear()
            self.add_bot_message("Chat cleared. How can I help you?")
    
    def show_about(self):
        QMessageBox.about(
            self,
            "About ML/NLP Chatbot",
            "ML/NLP Chatbot Desktop Application\n\n"
            "Version 1.0.0\n\n"
            "Advanced chatbot with intent classification\n"
            "and semantic retrieval capabilities.\n\n"
            "Built with PyQt5, TensorFlow, and FastAPI"
        )

def main():
    app = QApplication(sys.argv)
    window = ChatbotWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()