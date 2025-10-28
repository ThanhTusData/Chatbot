from tensorflow import keras
from pathlib import Path

def get_callbacks(output_dir: Path):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "checkpoint.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1
        )
    ]