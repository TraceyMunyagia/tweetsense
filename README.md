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

### Full pipeline

Run everything with one command:

```bash
python src/main.py
```

This command:

1. Generates 1,000 synthetic tech tweets in `data/raw/raw_tweets.csv`
2. Cleans and preprocesses the tweet text
3. Applies VADER and TextBlob sentiment analysis
4. Saves labeled data to `data/processed/labeled_tweets.csv`
5. Generates charts in `results/figures/`
6. Writes a summary report to `results/reports/summary.txt`

### Run each step manually

Generate raw tweets:

```bash
python src/fetch_tweets.py
```

Clean and preprocess tweets:

```bash
python src/data_preprocessing.py
```

Run sentiment analysis:

```bash
python src/sentimental_model.py
```

Generate visualizations and the summary report:

```bash
python src/visualization.py
```

---

## Streamlit Dashboard

Start the dashboard with:

```bash
streamlit run app.py
```

The dashboard reads `data/processed/labeled_tweets.csv`. If labeled data does not exist yet, run:

```bash
python src/main.py
```

You can also run the full pipeline from the dashboard sidebar.

---

## Outputs

After running the full pipeline, the main output files are:

```text
data/raw/raw_tweets.csv
data/processed/clean_tweets.csv
data/processed/labeled_tweets.csv
results/reports/summary.txt
results/figures/sentiment_bar.png
results/figures/sentiment_pie.png
results/figures/sentiment_trend.png
results/figures/top_words.png
results/figures/wordcloud_positive.png
results/figures/wordcloud_neutral.png
results/figures/wordcloud_negative.png
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
