"""
data_preprocessing.py
Day 2 – Data Cleaning & Preprocessing.

Functions:
    load_and_clean_data()  → returns a cleaned pandas DataFrame
"""

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_tweets.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_tweets.csv")


# ─── NLTK setup ───────────────────────────────────────────────────────────────
def _download_nltk_resources():
    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_download_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))


# ─── Cleaning helpers ─────────────────────────────────────────────────────────
def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text: str) -> str:
    """Remove the # symbol but keep the word."""
    return re.sub(r"#(\w+)", r"\1", text)


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single tweet."""
    if not isinstance(text, str):
        return ""
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_emojis(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)
    return text


def tokenize_and_remove_stopwords(text: str) -> str:
    """Tokenize, remove stopwords, return space-joined tokens."""
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(filtered)


def word_count(text: str) -> int:
    return len(text.split())


# ─── Main pipeline ────────────────────────────────────────────────────────────
def load_and_clean_data() -> pd.DataFrame | None:
    """
    Load raw tweets CSV, apply full cleaning pipeline, save processed CSV.
    Returns cleaned DataFrame (or None if raw file not found).
    """
    if not os.path.exists(RAW_PATH):
        print(f"[WARN] Raw data not found at {RAW_PATH}")
        return None

    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Loaded {len(df)} raw tweets.")

    # Drop rows with missing content
    df.dropna(subset=["content"], inplace=True)
    df.drop_duplicates(subset=["content"], inplace=True)

    # Clean text
    df["cleaned_text"] = df["content"].apply(clean_text)
    df["processed_text"] = df["cleaned_text"].apply(tokenize_and_remove_stopwords)

    # Drop empty rows after cleaning
    df = df[df["cleaned_text"].str.strip() != ""]
    df = df[df["processed_text"].str.strip() != ""]

    # Feature: tweet length
    df["word_count"] = df["cleaned_text"].apply(word_count)

    # Convert date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Save
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[INFO] Saved {len(df)} cleaned tweets → {PROCESSED_PATH}")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    if df is not None:
        print(df[["content", "cleaned_text", "processed_text", "word_count"]].head(5).to_string())
