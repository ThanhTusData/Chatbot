import logging
from typing import Dict

import spacy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from config import Config

class NLPProcessor:
    """Xử lý ngôn ngữ tự nhiên với spaCy và BERT"""
    
    def __init__(self, config: Config):
        self.config = config
        self.nlp = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.load_models()
        
    def load_models(self):
        """Tải các model NLP"""
        try:
            # Load spaCy model
            self.nlp = spacy.load(self.config.SPACY_MODEL)
            
            # Load BERT model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.config.BERT_MODEL)
            self.bert_model = AutoModel.from_pretrained(self.config.BERT_MODEL)
            
            logging.info("NLP models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading NLP models: {e}")
            
    def preprocess_text(self, text: str) -> Dict:
        """Tiền xử lý văn bản với spaCy"""
        doc = self.nlp(text.lower().strip())
        
        # Lemmatization và loại bỏ stopwords
        lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # POS tagging
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            'original': text,
            'lemmatized': lemmatized,
            'entities': entities,
            'pos_tags': pos_tags,
            'processed_text': ' '.join(lemmatized)
        }
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Tạo embeddings với BERT"""
        inputs = self.bert_tokenizer(text, return_tensors='pt', 
                                   max_length=self.config.MAX_SEQUENCE_LENGTH, 
                                   truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Lấy [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embeddings.flatten()