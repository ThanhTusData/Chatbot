
# 🤖 Chatbot ML NLP

An intelligent chatbot system leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify intents, generate responses, and support interactions through Web, Desktop, and Streamlit interfaces.

---

## 🎯 Project Objectives

- Classify user intents from input messages
- Generate contextual responses based on conversation history
- Support multi-platform deployment: Flask Web, Desktop App, Streamlit UI
- Integrate voice processing (speech-to-text / text-to-speech)

---

## 🧩 System Architecture

```
+-------------------+      +----------------+      +------------------+
| User Input (Text/ | ---> | Intent         | ---> | Response         |
| Voice via Web UI) |      | Classifier     |      | Generator        |
+-------------------+      +----------------+      +------------------+
        |                            |                         |
        |                            v                         |
        |                     +-------------+                 |
        |                     | Conversation|                 |
        |                     |  History    |<----------------+
        v                     +-------------+
+-------------------------+
| Output to Web / Desktop |
+-------------------------+
```

---

## 📂 Project Structure

```
chatbot_ml_nlp/
├── src/
│   ├── classification/
│   │   └── intent_classifier.py       # Intent classification logic
│   ├── core/
│   │   ├── chatbot.py                 # Central chatbot logic
│   │   └── conversation_history.py    # Conversation state tracking
│   ├── nlp/
│   │   └── processor.py               # NLP preprocessing and pipeline
│   ├── response/
│   │   └── response_generator.py      # Generates bot responses
│   ├── voice/
│   │   └── voice_processor.py         # Speech processing module
│   └── web/
│       └── templates/
│           └── chat.html              # Web interface (HTML template)
├── tests/                             # Unit and integration tests
├── flask_app.py                       # Flask app for chatbot API
├── desktop_app.py                     # Desktop GUI app
├── streamlit_app.py                   # Streamlit interface
├── main.py                            # Unified entry point (optional)
├── config.py                          # App configuration
├── deploy.sh / deploy.ps1            # Deployment scripts
├── docker-compose.yaml               # Docker orchestration
├── Dockerfile                        # Docker image build config
└── requirements.txt                  # Python dependencies
```

---

## 🚀 How to Run the Project

### 1. Install dependencies (locally)
```bash
pip install -r requirements.txt
```

### 2. Run with Streamlit (quick UI)
```bash
python streamlit_app.py
```

### 3. Run with Flask (web app)
```bash
python flask_app.py
```
Access via: [http://localhost:5000](http://localhost:5000)

### 4. Run the desktop interface
```bash
python desktop_app.py
```

### 5. Run with Docker
```bash
docker-compose up --build
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🔧 Extensions

- Multi-language support
- Integration with LLMs (ChatGPT, Claude, Gemini, etc.)
- Persistent conversation memory
- Integration with knowledge base or database