import streamlit as st
import requests
from datetime import datetime
import json
from config.config import config

st.set_page_config(
    page_title="ML/NLP Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", f"http://{config.API_HOST}:{config.API_PORT}")
    top_k = st.slider("Top K Predictions", 1, 10, 3)
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"Status: {health_data['status']}")
            st.info(f"Version: {health_data['version']}")
        else:
            st.error("API Unavailable")
    except:
        st.error("Cannot connect to API")

# Main content
st.title("🤖 ML/NLP Chatbot Demo")
st.markdown("Advanced chatbot with intent classification and semantic retrieval")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Intent Analysis", "📚 Document Search"])

# Chat Tab
with tab1:
    st.subheader("Interactive Chat")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "intent" in message:
                st.caption(f"Intent: {message['intent']} (confidence: {message['confidence']:.2f})")
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        try:
            response = requests.post(
                f"{api_url}/chat",
                json={"message": prompt},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_message = {
                    "role": "assistant",
                    "content": data["response"],
                    "intent": data["intent"]["intent"],
                    "confidence": data["intent"]["confidence"]
                }
                st.session_state.messages.append(bot_message)
                
                with st.chat_message("assistant"):
                    st.write(data["response"])
                    st.caption(f"Intent: {data['intent']['intent']} (confidence: {data['intent']['confidence']:.2f})")
            else:
                st.error("Failed to get response from API")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Intent Analysis Tab
with tab2:
    st.subheader("Intent Classification Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=100)
    
    if st.button("Analyze Intent", type="primary"):
        if text_input:
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": text_input, "top_k": top_k},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown("### Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Processing Time", f"{data['processing_time']:.3f}s")
                    with col2:
                        st.metric("Top Intent", data['predictions'][0]['intent'])
                    
                    st.markdown("### Intent Probabilities")
                    for pred in data['predictions']:
                        st.progress(pred['confidence'], text=f"{pred['intent']}: {pred['confidence']:.2%}")
                    
                    # Visualization
                    import pandas as pd
                    import plotly.express as px
                    
                    df = pd.DataFrame(data['predictions'])
                    fig = px.bar(df, x='intent', y='confidence', 
                                title='Intent Confidence Distribution',
                                labels={'confidence': 'Confidence', 'intent': 'Intent'},
                                color='confidence',
                                color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to analyze text")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to analyze")

# Document Search Tab
with tab3:
    st.subheader("Semantic Document Search")
    
    query = st.text_input("Search query:")
    search_top_k = st.slider("Number of results", 1, 20, 5, key="search_k")
    search_threshold = st.slider("Minimum score", 0.0, 1.0, threshold, key="search_threshold")
    
    if st.button("Search", type="primary"):
        if query:
            try:
                response = requests.post(
                    f"{api_url}/retrieve",
                    json={
                        "query": query,
                        "top_k": search_top_k,
                        "threshold": search_threshold
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['results']:
                        st.success(f"Found {len(data['results'])} relevant documents")
                        
                        for i, result in enumerate(data['results'], 1):
                            with st.expander(f"📄 Result {i} (Score: {result['score']:.3f})"):
                                st.write(result['content'])
                                if result.get('metadata'):
                                    st.json(result['metadata'])
                    else:
                        st.info("No documents found matching your query")
                else:
                    st.error("Search failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using TensorFlow, BERT, FastAPI, and Streamlit")