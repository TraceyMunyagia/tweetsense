import streamlit as st
import pandas as pd
import subprocess
import sys
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TweetSense — Tech Sentiment",
    page_icon="🧠",
    layout="wide",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
RAW_CSV      = BASE_DIR / "data" / "raw"        / "raw_tweets.csv"
LABELED_CSV  = BASE_DIR / "data" / "processed"  / "labeled_tweets.csv"
FIGURES_DIR  = BASE_DIR / "results" / "figures"
REPORT_PATH  = BASE_DIR / "results" / "reports"  / "summary.txt"

# ── Helpers ─────────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#1D9E75",
    "neutral":  "#7F77DD",
    "negative": "#D85A30",
}

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    # Normalise sentiment column to lowercase
    for col in ("sentiment", "vader_sentiment", "label"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df = df.rename(columns={col: "sentiment"})
            break
    # Normalise date column
    for col in ("date", "created_at", "timestamp", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.rename(columns={col: "date"})
            break
    return df

def sentiment_badge(label: str) -> str:
    colors = {"positive": "#E1F5EE", "negative": "#FAECE7", "neutral": "#EEEDFE"}
    text   = {"positive": "#0F6E56", "negative": "#993C1D", "neutral": "#534AB7"}
    bg  = colors.get(label, "#F1EFE8")
    txt = text.get(label,  "#5F5E5A")
    return (
        f'<span style="background:{bg};color:{txt};padding:2px 10px;'
        f'border-radius:99px;font-size:12px;font-weight:500;">{label}</span>'
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline controls")

    run_pipeline = st.button("▶ Run full pipeline", use_container_width=True)
    if run_pipeline:
        with st.spinner("Running main.py …"):
            result = subprocess.run([sys.executable, "src/main.py"], cwd=BASE_DIR)
        if result.returncode == 0:
            st.success("Pipeline complete! Reload to refresh data.")
            st.cache_data.clear()
        else:
            st.error("Pipeline failed — check your terminal for errors.")

    st.divider()
    st.markdown("### 🔍 Filter tweets")
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )
    keyword = st.text_input("Keyword search", placeholder="e.g. OpenAI, GPU …")
    max_rows = st.slider("Max tweets to display", 10, 200, 50, step=10)

    st.divider()
    st.markdown("### 📂 Data files")
    st.caption(f"Raw tweets: `{RAW_CSV}`")
    st.caption(f"Labeled: `{LABELED_CSV}`")

# ── Main ─────────────────────────────────────────────────────────────────────────
st.title("🧠 TweetSense — Tech Sentiment")
st.caption("VADER + TextBlob sentiment analysis on tech tweets")

# Check labeled CSV exists
if not LABELED_CSV.exists():
    st.warning(
        "**No labeled data found.**  \n"
        "Run the pipeline first:\n"
        "```bash\npython src/main.py\n```\n"
        "or step-by-step:\n"
        "```bash\npython src/fetch_tweets.py\n"
        "python src/data_preprocessing.py\n"
        "python src/sentimental_model.py\n```"
    )
    st.stop()

df = load_data(LABELED_CSV)

if "sentiment" not in df.columns:
    st.error("Could not find a sentiment column in `labeled_tweets.csv`. "
             "Expected one of: `sentiment`, `vader_sentiment`, `label`.")
    st.stop()

# ── Metric cards ────────────────────────────────────────────────────────────────
counts = df["sentiment"].value_counts()
total  = len(df)
pos_pct = round(counts.get("positive", 0) / total * 100)
neg_pct = round(counts.get("negative", 0) / total * 100)
neu_pct = round(counts.get("neutral",  0) / total * 100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tweets analysed", f"{total:,}")
c2.metric("😊 Positive",  f"{pos_pct}%", f"{counts.get('positive', 0)} tweets")
c3.metric("😐 Neutral",   f"{neu_pct}%", f"{counts.get('neutral',  0)} tweets")
c4.metric("😠 Negative",  f"{neg_pct}%", f"{counts.get('negative', 0)} tweets")

st.divider()

# ── Charts row ───────────────────────────────────────────────────────────────────
col_bar, col_pie = st.columns(2)

with col_bar:
    st.markdown("#### Sentiment distribution")
    chart_data = (
        df["sentiment"]
        .value_counts()
        .reindex(["positive", "neutral", "negative"])
        .fillna(0)
        .reset_index()
    )
    chart_data.columns = ["Sentiment", "Count"]
    chart_data["Color"] = chart_data["Sentiment"].map(SENTIMENT_COLORS)
    st.bar_chart(chart_data.set_index("Sentiment")["Count"])

with col_pie:
    st.markdown("#### Share breakdown")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3))
        labels  = ["Positive", "Neutral", "Negative"]
        sizes   = [pos_pct, neu_pct, neg_pct]
        colors  = [SENTIMENT_COLORS["positive"], SENTIMENT_COLORS["neutral"], SENTIMENT_COLORS["negative"]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.axis("equal")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib for the pie chart: `pip install matplotlib`")

# ── Trend chart ──────────────────────────────────────────────────────────────────
if "date" in df.columns and df["date"].notna().sum() > 0:
    st.markdown("#### 📈 Sentiment trend over time")
    trend = (
        df.dropna(subset=["date"])
        .set_index("date")
        .groupby([pd.Grouper(freq="D"), "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    )
    trend.columns = ["Positive", "Neutral", "Negative"]
    st.line_chart(trend, color=[
        SENTIMENT_COLORS["positive"],
        SENTIMENT_COLORS["neutral"],
        SENTIMENT_COLORS["negative"],
    ])
    st.divider()

# ── Saved figures ────────────────────────────────────────────────────────────────
figures = list(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
if figures:
    st.markdown("#### 🖼 Generated figures")
    cols = st.columns(min(len(figures), 3))
    for i, fig_path in enumerate(sorted(figures)):
        with cols[i % 3]:
            st.image(str(fig_path), caption=fig_path.stem.replace("_", " "), use_column_width=True)
    st.divider()

# ── Tweet feed ───────────────────────────────────────────────────────────────────
st.markdown("#### 🐦 Classified tweets")

filtered = df[df["sentiment"].isin(sentiment_filter)].copy()
if keyword.strip():
    text_col = next((c for c in ("tweet", "text", "cleaned_text", "processed_text", "content") if c in filtered.columns), None)
    if text_col:
        filtered = filtered[filtered[text_col].str.contains(keyword, case=False, na=False)]

text_col = next((c for c in ("tweet", "text", "cleaned_text", "processed_text", "content") if c in filtered.columns), None)

if filtered.empty:
    st.info("No tweets match your current filters.")
else:
    display_cols = [c for c in [text_col, "date", "sentiment"] if c]
    subset = filtered[display_cols].head(max_rows).reset_index(drop=True)

    if text_col:
        for _, row in subset.iterrows():
            badge = sentiment_badge(row["sentiment"])
            date_str = row["date"].strftime("%d %b %Y") if "date" in row and pd.notna(row.get("date")) else ""
            st.markdown(
                f'<div style="padding:10px 0; border-bottom:0.5px solid rgba(128,128,128,0.2);">'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
                f'<span style="font-size:12px; color:gray;">{date_str}</span>{badge}'
                f'</div>'
                f'<p style="margin:0; font-size:14px; line-height:1.6;">{row[text_col]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.dataframe(subset, use_container_width=True)

# ── Summary report ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📄 Summary report")
if REPORT_PATH.exists():
    with open(REPORT_PATH) as f:
        report_text = f.read()
    with st.expander("View summary.txt"):
        st.code(report_text, language="text")
    st.download_button(
        "⬇ Download report",
        data=report_text,
        file_name="summary.txt",
        mime="text/plain",
    )
else:
    st.caption("No report yet — run `python src/visualization.py` or the full pipeline.")
