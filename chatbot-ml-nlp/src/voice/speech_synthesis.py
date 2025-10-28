import pyttsx3
from typing import Optional

class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()
    
    def save_to_file(self, text: str, filename: str) -> None:
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()
    
    def set_voice(self, voice_index: int = 0) -> None:
        voices = self.engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
    
    def get_available_voices(self) -> list:
        voices = self.engine.getProperty('voices')
        return [(i, voice.name) for i, voice in enumerate(voices)]