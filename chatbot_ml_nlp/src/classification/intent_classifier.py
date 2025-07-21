from typing import List, Dict, Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from config import Config

class IntentClassifier:
    """Mô hình phân loại intent với LSTM/GRU"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intent_responses = {}
        
    def build_model(self, vocab_size: int, num_classes: int):
        """Xây dựng mô hình LSTM/GRU"""
        
        # Input layer
        input_layer = Input(shape=(self.config.MAX_SEQUENCE_LENGTH,))
        
        # Embedding layer
        embedding = Embedding(vocab_size, self.config.EMBEDDING_DIM, 
                            mask_zero=True)(input_layer)
        
        # LSTM layers với Attention
        lstm1 = LSTM(self.config.HIDDEN_UNITS, return_sequences=True, 
                    dropout=self.config.DROPOUT_RATE)(embedding)
        lstm2 = LSTM(self.config.HIDDEN_UNITS, return_sequences=True, 
                    dropout=self.config.DROPOUT_RATE)(lstm1)
        
        # Attention mechanism
        attention = Attention()([lstm2, lstm2])
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(attention)
        dropout = Dropout(self.config.DROPOUT_RATE)(dense1)
        
        # Output layer
        output = Dense(num_classes, activation='softmax')(dropout)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, training_data: List[Dict]):
        """Huấn luyện mô hình"""
        # Chuẩn bị dữ liệu
        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]
        
        # Tokenization
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.config.MAX_SEQUENCE_LENGTH)
        
        # Label encoding  
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded)
        
        # Build và train model
        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)
        
        self.build_model(vocab_size, num_classes)
        
        # Training với callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.model.fit(
            X, y,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Dự đoán intent và confidence"""
        if not self.model or not self.tokenizer:
            return "unknown", 0.0
            
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.config.MAX_SEQUENCE_LENGTH)
        
        # Predict
        prediction = self.model.predict(padded)[0]
        confidence = np.max(prediction)
        intent_idx = np.argmax(prediction)
        
        intent = self.label_encoder.inverse_transform([intent_idx])[0]
        
        return intent, confidence