"""
visualization.py
Day 4 – Visualization & Insights.

Generates:
    1. Sentiment distribution bar chart
    2. Sentiment pie chart
    3. Word clouds per sentiment class
    4. Sentiment trend over time (if timestamps available)
    5. Top words per sentiment (combined bar chart)

All figures saved to results/figures/.
Short summary report saved to results/reports/summary.txt.
"""

import os
from collections import Counter
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "results", "reports")
MPLCONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib")

os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "neutral":  "#2196F3",
    "negative": "#F44336",
}

sns.set_theme(style="whitegrid", palette="muted")


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── 1. Bar chart ─────────────────────────────────────────────────────────────
def plot_sentiment_bar(df: pd.DataFrame):
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Sentiment Distribution of Tech Tweets", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Tweets", fontsize=12)
    ax.set_ylim(0, max(counts.max() * 1.15, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(FIGURES_DIR, "sentiment_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")


# ─── 2. Pie chart ─────────────────────────────────────────────────────────────
def plot_sentiment_pie(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    labels = counts.index.tolist()
    colors = [SENTIMENT_COLORS.get(l, "#999") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax.set_title("Sentiment Share — Tech Tweets", fontsize=14, fontweight="bold", pad=16)

    path = os.path.join(FIGURES_DIR, "sentiment_pie.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")


# ─── 3. Word clouds ───────────────────────────────────────────────────────────
def plot_wordclouds(df: pd.DataFrame):
    text_col = "processed_text" if "processed_text" in df.columns else "cleaned_text"

    for sentiment in ["positive", "negative", "neutral"]:
        subset = df[df["sentiment"] == sentiment]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        ax.set_title(
            f"Most Common Words — {sentiment.capitalize()} Tweets",
            fontsize=14, fontweight="bold", pad=12,
        )

        text = " ".join(subset[text_col].dropna().tolist()) if not subset.empty else ""
        if text.strip():
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="RdYlGn" if sentiment != "negative" else "Reds",
                max_words=100,
                collocations=False,
            ).generate(text)
            ax.imshow(wc, interpolation="bilinear")
        else:
            ax.text(
                0.5,
                0.5,
                f"No {sentiment} tweets available",
                ha="center",
                va="center",
                fontsize=16,
                color="#666666",
            )

        path = os.path.join(FIGURES_DIR, f"wordcloud_{sentiment}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved → {path}")


# ─── 4. Sentiment trend over time ─────────────────────────────────────────────
def plot_sentiment_trend(df: pd.DataFrame):
    if "date" not in df.columns:
        print("[INFO] No 'date' column — skipping trend chart.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df.dropna(subset=["date"], inplace=True)

    if df.empty:
        return

    df["day"] = df["date"].dt.date
    trend = (
        df.groupby(["day", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="day", columns="sentiment", values="count")
        .fillna(0)
    )

    if trend.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for sentiment, color in SENTIMENT_COLORS.items():
        if sentiment in trend.columns:
            ax.plot(trend.index, trend[sentiment], label=sentiment.capitalize(),
                    color=color, linewidth=2, marker="o", markersize=4)

    ax.set_title("Sentiment Trend Over Time", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Tweet Count", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(FIGURES_DIR, "sentiment_trend.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")


# ─── 5. Top words per sentiment ───────────────────────────────────────────────
def plot_top_words(df: pd.DataFrame, top_n: int = 15):
    text_col = "processed_text" if "processed_text" in df.columns else "cleaned_text"
    sentiments = ["positive", "negative", "neutral"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False)
    has_data = False

    for ax, sentiment in zip(axes, sentiments):
        subset = df[df["sentiment"] == sentiment]
        if subset.empty:
            ax.axis("off")
            ax.set_title(f"{sentiment.capitalize()} Tweets", fontsize=13, fontweight="bold")
            continue

        words = " ".join(subset[text_col].dropna()).split()
        freq = Counter(words).most_common(top_n)
        if not freq:
            ax.axis("off")
            ax.set_title(f"{sentiment.capitalize()} Tweets", fontsize=13, fontweight="bold")
            continue

        terms, counts = zip(*freq)
        has_data = True
        color = SENTIMENT_COLORS[sentiment]
        ax.barh(list(reversed(terms)), list(reversed(counts)), color=color, edgecolor="white")
        ax.set_title(f"{sentiment.capitalize()} Tweets", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if not has_data:
        plt.close(fig)
        print("[INFO] No words available — skipping top words chart.")
        return

    fig.suptitle(f"Top {top_n} Words by Sentiment", fontsize=15, fontweight="bold")
    path = os.path.join(FIGURES_DIR, "top_words.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {path}")


# ─── 6. Summary report ────────────────────────────────────────────────────────
def write_summary_report(df: pd.DataFrame):
    total = len(df)
    counts = df["sentiment"].value_counts()
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral", 0)

    avg_vader = df["vader_score"].mean() if "vader_score" in df.columns else None

    report_lines = [
        "=" * 60,
        "  AI Sentiment Analysis — Tech Tweets",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "DATASET OVERVIEW",
        f"  Total tweets analyzed : {total}",
        f"  Positive              : {pos} ({pos/total*100 if total else 0:.1f}%)",
        f"  Neutral               : {neu} ({neu/total*100 if total else 0:.1f}%)",
        f"  Negative              : {neg} ({neg/total*100 if total else 0:.1f}%)",
    ]

    if avg_vader is not None:
        report_lines.append(f"  Avg VADER compound    : {avg_vader:.4f}")

    report_lines += [
        "",
        "KEY FINDINGS",
        f"  - The dominant sentiment is '{counts.idxmax() if not counts.empty else 'N/A'}'.",
    ]

    if avg_vader is not None:
        if avg_vader > 0.05:
            report_lines.append("  - Overall tone toward AI/tech is positive.")
        elif avg_vader < -0.05:
            report_lines.append("  - Overall tone toward AI/tech is negative.")
        else:
            report_lines.append("  - Tweets show a largely mixed/neutral tone.")

    report_lines += [
        "",
        "OUTPUT FILES",
        "  results/figures/sentiment_bar.png",
        "  results/figures/sentiment_pie.png",
        "  results/figures/wordcloud_positive.png",
        "  results/figures/wordcloud_neutral.png",
        "  results/figures/wordcloud_negative.png",
        "  results/figures/sentiment_trend.png",
        "  results/figures/top_words.png",
        "  data/processed/labeled_tweets.csv",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    path = os.path.join(REPORTS_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(report)

    print(f"[INFO] Report saved → {path}")
    print("\n" + report)


# ─── Entry ────────────────────────────────────────────────────────────────────
def generate_all_visualizations(df: pd.DataFrame):
    _ensure_dirs()
    plot_sentiment_bar(df)
    plot_sentiment_pie(df)
    plot_wordclouds(df)
    plot_sentiment_trend(df)
    plot_top_words(df)
    write_summary_report(df)


if __name__ == "__main__":
    labeled_path = os.path.join(BASE_DIR, "data", "processed", "labeled_tweets.csv")
    if not os.path.exists(labeled_path):
        print(f"[ERROR] Run sentimental_model.py first. {labeled_path} not found.")
    else:
        df = pd.read_csv(labeled_path)
        generate_all_visualizations(df)
