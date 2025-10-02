# src/data/prepare.py
import json
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: str):
    """Load training_data.json (list of {text,intent}) -> pandas.DataFrame"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("training data must contain 'text' and 'intent' fields")
    return df


def clean_text(s: str) -> str:
    """Basic cleaning: strip, lower, collapse spaces"""
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to text column and drop empty"""
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""].reset_index(drop=True)
    return df


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["intent"])
    return train.reset_index(drop=True), test.reset_index(drop=True)
