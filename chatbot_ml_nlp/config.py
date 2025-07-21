class Config:
    """Cấu hình toàn cục cho chatbot"""
    
    # Model configuration
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 300
    HIDDEN_UNITS = 128
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # NLP Configuration
    SPACY_MODEL = "en_core_web_sm"  # hoặc "vi_core_news_sm" cho tiếng Việt
    BERT_MODEL = "bert-base-uncased"
    CONFIDENCE_THRESHOLD = 0.7
    
    # File paths
    MODEL_DIR = "models"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    
    # Supported languages
    LANGUAGES = {
        'en': 'English',
        'vi': 'Vietnamese',
        'es': 'Spanish',
        'fr': 'French'
    }