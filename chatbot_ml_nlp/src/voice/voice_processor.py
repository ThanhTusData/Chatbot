import logging
import speech_recognition as sr
import pyttsx3

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