from flask import Blueprint, render_template, request, jsonify, session
import requests
from datetime import datetime

bp = Blueprint('main', __name__)

API_BASE_URL = None  # Will be set by app

def init_routes(api_url):
    global API_BASE_URL
    API_BASE_URL = api_url

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat')
def chat():
    return render_template('chat.html')

@bp.route('/history')
def history():
    chat_history = session.get('chat_history', [])
    return render_template('history.html', history=chat_history)

@bp.route('/api/message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={'message': message},
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
            
            return jsonify(result)
        else:
            return jsonify({'error': 'API error'}), response.status_code
    
    except Exception as e:
        return jsonify({'error': str(e)}), 503