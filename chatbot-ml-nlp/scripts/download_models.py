#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def download_spacy_models():
    print("Downloading spaCy models...")
    models = ["en_core_web_sm", "en_core_web_md"]
    
    for model in models:
        print(f"Downloading {model}...")
        subprocess.run(["python", "-m", "spacy", "download", model])

def download_sentence_transformers():
    print("Downloading sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ]
    
    for model in models:
        print(f"Downloading {model}...")
        SentenceTransformer(model)
        print(f"✓ {model} downloaded")

def main():
    print("=" * 60)
    print("Downloading Required Models")
    print("=" * 60)
    
    download_spacy_models()
    download_sentence_transformers()
    
    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()