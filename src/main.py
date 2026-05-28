
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from fetch_tweets import main as collect_tweets
from data_preprocessing import load_and_clean_data
from sentimental_model import run_sentiment_analysis
from visualization import generate_all_visualizations
from utils import ensure_dirs, log

def main():
    log("=== AI Sentiment Analysis on Tech Tweets ===")
    ensure_dirs()

    log("Step 1: Collecting tweets...")
    collect_tweets()

    log("Step 2: Loading and preprocessing data...")
    df = load_and_clean_data()
    if df is None or df.empty:
        log("ERROR: No data found after collection.")
        sys.exit(1)

    log("Step 3: Running sentiment analysis...")
    df = run_sentiment_analysis(df)

    log("Step 4: Generating visualizations and report...")
    generate_all_visualizations(df)

    log("=== Pipeline complete! ===")
    log("Results saved to results/figures/ and results/reports/")

if __name__ == "__main__":
    main()
