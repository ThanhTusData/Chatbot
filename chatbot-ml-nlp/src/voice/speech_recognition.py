import speech_recognition as sr
from typing import Optional

class SpeechToText:
    def __init__(self, engine: str = "google"):
        self.recognizer = sr.Recognizer()
        self.engine = engine
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = self.recognizer.listen(source, timeout=timeout)
                print("Processing...")
                
                if self.engine == "google":
                    text = self.recognizer.recognize_google(audio)
                elif self.engine == "sphinx":
                    text = self.recognizer.recognize_sphinx(audio)
                else:
                    raise ValueError(f"Unknown engine: {self.engine}")
                
                return text
            except sr.WaitTimeoutError:
                print("No speech detected")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Error: {e}")
                return None
    
    def recognize_from_file(self, audio_file: str) -> Optional[str]:
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
            
            try:
                if self.engine == "google":
                    return self.recognizer.recognize_google(audio)
                elif self.engine == "sphinx":
                    return self.recognizer.recognize_sphinx(audio)
            except Exception as e:
                print(f"Error: {e}")
                return None