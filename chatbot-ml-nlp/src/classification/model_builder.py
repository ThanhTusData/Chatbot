import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple

class IntentClassifierBuilder:
    @staticmethod
    def build_lstm_attention_model(
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int,
        lstm_units: int = 128,
        dropout_rate: float = 0.5
    ) -> keras.Model:
        inputs = layers.Input(shape=(max_length,))
        
        x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
        x = layers.SpatialDropout1D(0.2)(x)
        
        # Bidirectional LSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        )(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(lstm_units * 2)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = layers.multiply([lstm_out, attention])
        sent_representation = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(sent_representation)
        
        # Classification layers
        x = layers.Dense(64, activation='relu')(sent_representation)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def build_cnn_model(
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int,
        num_filters: int = 128,
        filter_sizes: Tuple[int, ...] = (3, 4, 5)
    ) -> keras.Model:
        inputs = layers.Input(shape=(max_length,))
        x = layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        # Multiple CNN branches
        conv_blocks = []
        for filter_size in filter_sizes:
            conv = layers.Conv1D(num_filters, filter_size, activation='relu')(x)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        x = layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model