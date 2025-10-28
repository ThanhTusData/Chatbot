from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
import uuid
from datetime import datetime
import logging

from config.config import config

app = Flask(__name__)
app.secret_key = config.SECRET_KEY if hasattr(config, 'SECRET_KEY') else 'dev-secret-key'
CORS(app)

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

API_BASE_URL = f"http://{config.API_HOST}:{config.API_PORT}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html', session_id=session['session_id'])

@app.route('/history')
def history():
    chat_history = session.get('chat_history', [])
    return render_template('history.html', history=chat_history)

@app.route('/api/message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                'message': message,
                'session_id': session.get('session_id')
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'chat_history' not in session:
                session['chat_history'] = []
            
            session['chat_history'].append({
                'user': message,
                'bot': result['response'],
                'timestamp': datetime.now().isoformat()
            })
            session.modified = True
            
            return jsonify(result)
        else:
            return jsonify({'error': 'API error'}), response.status_code
    
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        return jsonify({'error': 'Failed to connect to API'}), 503

@app.route('/api/health')
def api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except requests.RequestException:
        return jsonify({'status': 'unhealthy'}), 503

def main():
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=config.DEBUG
    )

if __name__ == '__main__':
    main()