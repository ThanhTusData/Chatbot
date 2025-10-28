class VoiceHandler {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.setupRecognition();
        } else if ('SpeechRecognition' in window) {
            this.recognition = new SpeechRecognition();
            this.setupRecognition();
        }
    }
    
    setupRecognition() {
        if (!this.recognition) return;
        
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.onSpeechResult(transcript);
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            this.onError(event.error);
        };
        
        this.recognition.onend = () => {
            this.isListening = false;
            this.onListeningEnd();
        };
    }
    
    startListening() {
        if (!this.recognition) {
            alert('Speech recognition not supported in this browser');
            return;
        }
        
        if (this.isListening) return;
        
        this.isListening = true;
        this.recognition.start();
        this.onListeningStart();
    }
    
    stopListening() {
        if (!this.recognition || !this.isListening) return;
        
        this.recognition.stop();
        this.isListening = false;
    }
    
    speak(text) {
        if (!this.synthesis) {
            console.warn('Speech synthesis not supported');
            return;
        }
        
        // Cancel any ongoing speech
        this.synthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        this.synthesis.speak(utterance);
    }
    
    // Callbacks (to be overridden)
    onSpeechResult(transcript) {
        console.log('Speech result:', transcript);
    }
    
    onListeningStart() {
        console.log('Listening started');
    }
    
    onListeningEnd() {
        console.log('Listening ended');
    }
    
    onError(error) {
        console.error('Error:', error);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceHandler;
}