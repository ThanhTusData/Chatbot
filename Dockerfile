FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    portaudio19-dev \\
    python3-pyaudio \\
    espeak \\
    espeak-data \\
    libespeak1 \\
    libespeak-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p models data logs

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]