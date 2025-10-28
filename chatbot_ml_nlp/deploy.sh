#!/bin/bash

# Deployment script for Intelligent Chatbot

echo "🚀 Deploying Intelligent Chatbot..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download vi_core_news_sm

# Create directories
mkdir -p models data logs templates static

# Set environment variables
export FLASK_APP=main.py
export FLASK_ENV=production

# Run database migrations (if any)
# python migrate.py

# Start the application
echo "✅ Starting chatbot server..."
python main.py

echo "🎉 Deployment completed!"
echo "🌐 Visit: http://localhost:5000"