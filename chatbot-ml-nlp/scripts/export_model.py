#!/usr/bin/env python3
import argparse
import tensorflow as tf
from pathlib import Path
import json

def export_to_saved_model(model_path: str, export_path: str):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Exporting to {export_path}")
    model.export(export_path)
    
    print("Model exported successfully!")

def export_to_tflite(model_path: str, export_path: str):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    print(f"Saving TFLite model to {export_path}")
    with open(export_path, 'wb') as f:
        f.write(tflite_model)
    
    print("TFLite model exported successfully!")

def export_to_onnx(model_path: str, export_path: str):
    try:
        import tf2onnx
        import onnx
    except ImportError:
        print("Error: tf2onnx not installed. Run: pip install tf2onnx")
        return
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to ONNX
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
    
    print(f"Converting to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
    
    print(f"Saving ONNX model to {export_path}")
    onnx.save(model_proto, export_path)
    
    print("ONNX model exported successfully!")

def main():
    parser = argparse.ArgumentParser(description='Export trained model to different formats')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--format', type=str, choices=['saved_model', 'tflite', 'onnx'], 
                       default='saved_model', help='Export format')
    
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'saved_model':
        export_to_saved_model(args.model, args.output)
    elif args.format == 'tflite':
        export_to_tflite(args.model, args.output)
    elif args.format == 'onnx':
        export_to_onnx(args.model, args.output)

if __name__ == '__main__':
    main()