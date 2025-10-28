import spacy
import re
from typing import List, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class NLPProcessor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        self.bert_tokenizer = None
        self.bert_model = None
    
    def preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(self.preprocess_text(text))
        return [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    def lemmatize(self, text: str) -> List[str]:
        doc = self.nlp(self.preprocess_text(text))
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_bert_embeddings(self, text: str, model_name: str = "bert-base-uncased") -> np.ndarray:
        if self.bert_tokenizer is None:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
        
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()