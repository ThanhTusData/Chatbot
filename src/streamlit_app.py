import streamlit as st
from config import Config
from src.core.chatbot import IntelligentChatbot

def create_streamlit_app():
    """Tạo ứng dụng Streamlit làm giao diện thay thế"""
    
    st.set_page_config(
        page_title="Intelligent Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 70%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background-color: white;
        color: #333;
        border: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Intelligent Chatbot")
    st.markdown("*Powered by ML/NLP • Hỗ trợ đa ngôn ngữ • Nhận diện giọng nói*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Language selection
        language = st.selectbox(
            "Ngôn ngữ:",
            options=['vi', 'en', 'es', 'fr'],
            format_func=lambda x: {
                'vi': '🇻🇳 Tiếng Việt',
                'en': '🇺🇸 English', 
                'es': '🇪🇸 Español',
                'fr': '🇫🇷 Français'
            }[x]
        )
        
        # Voice options
        use_voice = st.checkbox("🎤 Sử dụng giọng nói")
        
        # Model info
        st.header("📊 Thông tin Model")
        st.info("""
        - **NLP**: spaCy + BERT embeddings
        - **ML**: LSTM/GRU với Attention
        - **Framework**: TensorFlow 2.x
        - **Voice**: Speech Recognition
        """)
        
        # Clear history
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chatbot = None
    
    # Initialize chatbot
    if st.session_state.chatbot is None:
        with st.spinner("Đang khởi tạo chatbot..."):
            config = Config()
            st.session_state.chatbot = IntelligentChatbot(config)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show metadata for bot messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Intent: {metadata.get('intent', 'unknown')}")
                with col2:
                    confidence = metadata.get('confidence', 0)
                    st.caption(f"Confidence: {confidence:.2%}")
    
    # Chat input
    if prompt := st.chat_input("Nhập tin nhắn của bạn..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                result = st.session_state.chatbot.process_message(prompt, use_voice)
                
                st.write(result['bot_response'])
                
                # Show metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Intent: {result['intent']}")
                with col2:
                    st.caption(f"Confidence: {result['confidence']:.2%}")
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['bot_response'],
                    "metadata": result
                })