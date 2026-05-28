# 🧠 AI-Powered Sentiment Analysis on Tech Tweets

Classify tech-related tweets as **positive**, **negative**, or **neutral** using VADER and TextBlob — no Twitter API key required.

---

## 📁 Project Structure

```
AI Sentimental Analysis in Tech Tweets/
├── data/
│   ├── raw/               ← raw tweets collected by fetch_tweets.py
│   └── processed/         ← cleaned & labeled CSVs
├── notebooks/             ← optional Jupyter notebooks for exploration
├── results/
│   ├── figures/           ← all charts and word clouds
│   └── reports/           ← summary.txt report
├── src/
│   ├── fetch_tweets.py    ← Day 1: scrape tweets with snscrape
│   ├── data_preprocessing.py  ← Day 2: clean & preprocess text
│   ├── sentimental_model.py   ← Day 3: VADER + TextBlob sentiment
│   ├── visualization.py   ← Day 4: charts, word clouds, report
│   ├── utils.py           ← shared helpers
│   └── main.py            ← Day 5: end-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 🚀 Running the Project

The repository includes a small seed `data/raw/raw_tweets.csv` so the pipeline can be verified offline. Running `python src/main.py` will try live collection first, then continue with the existing raw CSV if `snscrape` cannot reach Twitter/X from your environment.

### Optional dashboard
```bash
streamlit run app.py
```

---

## 📊 Sample Results

| Sentiment | Count | % |
|-----------|-------|---|
| Positive  | ~420  | ~44% |
| Neutral   | ~330  | ~34% |
| Negative  | ~210  | ~22% |

**Output figures:**
- `sentiment_bar.png` — distribution bar chart
- `sentiment_pie.png` — sentiment share pie chart
- `wordcloud_positive/neutral/negative.png` — word clouds per class
- `sentiment_trend.png` — daily trend over time
- `top_words.png` — combined top 15 words per sentiment

---

## 🛠 Tools & Libraries

| Purpose | Library |
|---------|---------|
| Data handling | `pandas`, `numpy` |
| Text processing | `nltk`, `re` |
| Sentiment analysis | `vaderSentiment`, `textblob` |
| Tweet collection | `snscrape` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Optional ML | `scikit-learn` |

---

## 💡 Why snscrape?

- ✅ No API key or approval required
- ✅ Works immediately
- ✅ Great for learning projects (500–1,000 tweets)
- ✅ Simple CLI interface

---

## 📝 License

For educational / portfolio use only.
