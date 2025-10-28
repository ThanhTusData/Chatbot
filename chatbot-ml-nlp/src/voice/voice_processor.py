from voice.speech_recognition import SpeechToText
from voice.speech_synthesis import TextToSpeech
from typing import Optional

class VoiceProcessor:
    def __init__(self, stt_engine: str = "google", tts_rate: int = 150):
        self.stt = SpeechToText(engine=stt_engine)
        self.tts = TextToSpeech(rate=tts_rate)
    
    def listen_and_respond(self, response_callback) -> Optional[str]:
        # Listen for user input
        user_text = self.stt.listen()
        
        if user_text:
            print(f"User said: {user_text}")
            
            # Get response from callback
            response = response_callback(user_text)
            
            # Speak response
            if response:
                self.tts.speak(response)
            
            return user_text
        
        return None
    
    def process_audio_file(self, audio_file: str, response_callback) -> Optional[str]:
        text = self.stt.recognize_from_file(audio_file)
        
        if text:
            response = response_callback(text)
            return response
        
        return None