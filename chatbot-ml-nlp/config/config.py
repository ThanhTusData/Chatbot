import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    # Application
    APP_NAME: str = "ChatbotMLNLP"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 5000
    
    # Models
    INTENT_MODEL_PATH: str = "models/intent/model.h5"
    TOKENIZER_PATH: str = "models/intent/tokenizer.pkl"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH: int = 100
    EMBEDDING_DIM: int = 384
    
    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 5
    
    # Retrieval
    VECTOR_DB_TYPE: str = "faiss"
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    KB_INDEX_PATH: str = "indexes/kb/"
    
    # Voice
    ENABLE_VOICE: bool = False
    TTS_ENGINE: str = "pyttsx3"
    STT_ENGINE: str = "google"
    
    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 8001
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

config = Config()