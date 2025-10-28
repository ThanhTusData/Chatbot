import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

from training.data_loader import DataLoader
from nlp.tokenizer import CustomTokenizer
from classification.model_builder import IntentClassifierBuilder
from training.callbacks import get_callbacks

def train_intent_model(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_length: int = 100,
    embedding_dim: int = 128,
    lstm_units: int = 128
):
    print(f"Loading data from {data_path}")
    loader = DataLoader(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_data()
    
    print("Tokenizing texts")
    tokenizer = CustomTokenizer(max_length=max_length)
    tokenizer.fit(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    num_classes = len(np.unique(y_train))
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Building model with vocab_size={tokenizer.vocab_size}, num_classes={num_classes}")
    model = IntentClassifierBuilder.build_lstm_attention_model(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        num_classes=num_classes,
        lstm_units=lstm_units
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = get_callbacks(output_path)
    
    print("Training model")
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Evaluating on test set")
    test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    print("Saving model and artifacts")
    model.save(output_path / "model.h5")
    tokenizer.save(output_path / "tokenizer.pkl")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'embedding_dim': embedding_dim,
        'num_classes': num_classes,
        'label_encoder': loader.get_label_mapping(),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss)
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete. Model saved to {output_path}")
    return history, test_acc

def main():
    parser = argparse.ArgumentParser(description='Train intent classification model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='models/intent', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train_intent_model(args.data, args.output, args.epochs, args.batch_size, args.lr)

if __name__ == '__main__':
    main()