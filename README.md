# AI-Powered Sentiment Analysis on Tech Tweets

This project generates a synthetic tech-tweet dataset, cleans the text, classifies sentiment with VADER and TextBlob, creates visual reports, and optionally displays the results in a Streamlit dashboard.

The current version runs fully offline. No Twitter/X API key or live scraping is required.

---

## Project Structure

```text
AI_Sentiment_Analysis/
├── app.py                         # Streamlit dashboard
├── data/
│   ├── raw/
│   │   └── raw_tweets.csv         # Generated synthetic tweets
│   └── processed/
│       ├── clean_tweets.csv       # Cleaned/preprocessed tweets
│       └── labeled_tweets.csv     # Tweets with sentiment labels
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   ├── figures/                   # Charts and word clouds
│   └── reports/
│       └── summary.txt            # Text summary report
├── src/
│   ├── fetch_tweets.py            # Generates synthetic tech tweets
│   ├── data_preprocessing.py      # Cleans and tokenizes tweet text
│   ├── sentimental_model.py       # VADER + TextBlob sentiment analysis
│   ├── visualization.py           # Figures and report generation
│   ├── utils.py                   # Shared helpers
│   └── main.py                    # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK resources are downloaded automatically by `src/data_preprocessing.py` on first run if they are missing.

---

## Run the Project

The repository includes a small seed `data/raw/raw_tweets.csv` so the pipeline can be verified offline. Running `python src/main.py` will try live collection first, then continue with the existing raw CSV if `snscrape` cannot reach Twitter/X from your environment.

### Day 1 — Collect tweets
```bash
python src/fetch_tweets.py
```
Fetches ~1,000 tech tweets using `snscrape` and saves them to `data/raw/raw_tweets.csv`.

### Day 2 — Clean data
```bash
python src/data_preprocessing.py
```
Removes URLs, mentions, emojis, stopwords. Saves to `data/processed/clean_tweets.csv`.

### Day 3 — Sentiment analysis
```bash
python src/sentimental_model.py
```
Applies VADER + TextBlob. Saves labeled data to `data/processed/labeled_tweets.csv`.

### Day 4 — Visualizations
```bash
python src/visualization.py
```
Generates 7 figure files and a text report. Saved to `results/`.

### Day 5 — Full pipeline (one command)
```bash
python src/main.py
```
Runs tweet collection, preprocessing, sentiment analysis, visualizations, and report generation.

To reuse an existing `data/raw/raw_tweets.csv` without scraping again:
```bash
python src/main.py --skip-fetch
```

### Optional dashboard
```bash
streamlit run app.py
```

---

## Methods

| Stage | Description |
| --- | --- |
| Data generation | Creates 1,000 synthetic tech tweets with positive, neutral, and negative examples |
| Cleaning | Removes URLs, mentions, emojis, punctuation, stopwords, duplicate text, and empty rows |
| Sentiment analysis | Uses VADER compound scores and TextBlob polarity/subjectivity |
| Final label | Uses VADER/TextBlob agreement; falls back to VADER when they disagree |
| Reporting | Produces charts, word clouds, time trends, top words, and a text summary |

---

## Tools and Libraries

| Purpose | Library |
| --- | --- |
| Data handling | `pandas`, `numpy` |
| Text processing | `nltk`, `re` |
| Sentiment analysis | `vaderSentiment`, `textblob` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Dashboard | `streamlit` |
| Optional ML | `scikit-learn` |

---

## Notes

- `src/main.py` always regenerates `data/raw/raw_tweets.csv` before processing.
- The generated tweet dates start from January 2024 and are used for the sentiment trend chart.
- `snscrape` is still listed in `requirements.txt`, but the current `fetch_tweets.py` implementation does not use live scraping.

---

## License

For educational and portfolio use.
