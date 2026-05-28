"""
sentimental_model.py
Day 3 – Sentiment Analysis using VADER and TextBlob.

Functions:
    run_sentiment_analysis(df)  → DataFrame with sentiment columns added
    sample_verification(df)     → prints sample tweets per sentiment class
    train_ml_model(df)          → optional scikit-learn logistic regression
"""

import os
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional ML
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_tweets.csv")
LABELED_PATH = os.path.join(BASE_DIR, "data", "processed", "labeled_tweets.csv")

analyzer = SentimentIntensityAnalyzer()


# ─── VADER ────────────────────────────────────────────────────────────────────
def vader_sentiment(text: str) -> tuple[float, str]:
    """Returns (compound_score, label)."""
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


# ─── TextBlob ─────────────────────────────────────────────────────────────────
def textblob_sentiment(text: str) -> tuple[float, float, str]:
    """Returns (polarity, subjectivity, label)."""
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.05:
        label = "positive"
    elif polarity < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return polarity, subjectivity, label


# ─── Ensemble ─────────────────────────────────────────────────────────────────
def ensemble_label(vader_lbl: str, tb_lbl: str) -> str:
    """
    Combine VADER and TextBlob labels.
    Agreement → use that label.
    Disagreement → default to VADER (more suited for social media).
    """
    if vader_lbl == tb_lbl:
        return vader_lbl
    return vader_lbl


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Apply VADER + TextBlob to cleaned_text. Adds columns and saves labeled CSV."""
    text_col = "cleaned_text" if "cleaned_text" in df.columns else "content"

    print("[INFO] Applying VADER sentiment analysis...")
    vader_results = df[text_col].apply(vader_sentiment)
    df["vader_score"] = vader_results.apply(lambda x: x[0])
    df["vader_label"] = vader_results.apply(lambda x: x[1])

    print("[INFO] Applying TextBlob sentiment analysis...")
    tb_results = df[text_col].apply(textblob_sentiment)
    df["tb_polarity"] = tb_results.apply(lambda x: x[0])
    df["tb_subjectivity"] = tb_results.apply(lambda x: x[1])
    df["tb_label"] = tb_results.apply(lambda x: x[2])

    print("[INFO] Computing ensemble label...")
    df["sentiment"] = df.apply(
        lambda row: ensemble_label(row["vader_label"], row["tb_label"]), axis=1
    )

    # Distribution summary
    dist = df["sentiment"].value_counts()
    print(f"\n[INFO] Sentiment distribution:\n{dist.to_string()}\n")

    # Save
    os.makedirs(os.path.dirname(LABELED_PATH), exist_ok=True)
    df.to_csv(LABELED_PATH, index=False)
    print(f"[INFO] Saved labeled tweets → {LABELED_PATH}")

    return df


# ─── Verification ─────────────────────────────────────────────────────────────
def sample_verification(df: pd.DataFrame, n: int = 3):
    """Print sample tweets for each sentiment class for manual review."""
    print("\n=== Sample Verification ===")
    for label in ["positive", "negative", "neutral"]:
        subset = df[df["sentiment"] == label]
        print(f"\n--- {label.upper()} (showing {min(n, len(subset))}) ---")
        for _, row in subset.head(n).iterrows():
            content = row.get("content", row.get("cleaned_text", ""))
            score = row.get("vader_score", "N/A")
            print(f"  Score: {score:.3f} | {content[:140]}")


# ─── Optional ML Model ────────────────────────────────────────────────────────
def train_ml_model(df: pd.DataFrame):
    """
    Optional: Train a Logistic Regression with TF-IDF features.
    Uses VADER labels as ground truth.
    """
    if not ML_AVAILABLE:
        print("[WARN] scikit-learn not available. Skipping ML model.")
        return

    text_col = "processed_text" if "processed_text" in df.columns else "cleaned_text"
    df = df.dropna(subset=[text_col, "sentiment"])

    X = df[text_col]
    y = df["sentiment"]

    if len(X) < 50:
        print("[WARN] Not enough data for ML model (need ≥50 samples).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print("\n[INFO] ML Model Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    if not os.path.exists(PROCESSED_PATH):
        print(f"[ERROR] Run data_preprocessing.py first. {PROCESSED_PATH} not found.")
    else:
        df = pd.read_csv(PROCESSED_PATH)
        df = run_sentiment_analysis(df)
        sample_verification(df)
        train_ml_model(df)
