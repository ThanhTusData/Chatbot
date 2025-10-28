#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_intent_data(input_file: Path, output_file: Path):
    print(f"Loading data from {input_file}")
    data = load_jsonl(input_file)
    
    processed_data = []
    for item in data:
        if 'text' in item and 'intent' in item:
            processed_item = {
                'text': clean_text(item['text']),
                'intent': item['intent'],
                'original_text': item['text']
            }
            processed_data.append(processed_item)
    
    print(f"Processed {len(processed_data)} items")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess training data')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    preprocess_intent_data(Path(args.input), Path(args.output))

if __name__ == '__main__':
    main()