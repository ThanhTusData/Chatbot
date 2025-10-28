import re
from typing import List, Set
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS: Set[str] = set(stopwords.words('english'))

def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_words)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    words = text.lower().split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 3]
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0