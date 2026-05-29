"""
utils/sentiment_analysis.py
============================
Sentiment analysis helpers for googleplaystore_user_reviews.csv.
Uses TextBlob for polarity when pre-labelled Sentiment column missing.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Common English stop words (no NLTK dependency)
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'he','him','his','she','her','hers','it','its','they','them','their','what',
    'which','who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','do','does','did','will','would','shall',
    'should','may','might','must','can','could','a','an','the','and','but','or',
    'nor','for','so','yet','at','by','in','of','on','to','up','as','into','with',
    'about','against','between','after','before','during','through','from','out',
    'very','just','app','apps','it','s','t','m','re','ve','ll','d','don','didn',
    'doesn','isn','wasn','aren','weren','hasn','haven','hadn','won','wouldn',
    'couldn','shouldn','get','got','one','use','also','even','well',
}


# ──────────────────────────────────────────────
# LOADING + BASIC CLEAN
# ──────────────────────────────────────────────

def load_reviews(path: str) -> pd.DataFrame:
    """Load reviews CSV, ensure required columns exist."""
    df = pd.read_csv(path)
    df.dropna(subset=['Translated_Review'], inplace=True)
    df['Translated_Review'] = df['Translated_Review'].astype(str)

    # Compute polarity if not present
    if 'Sentiment_Polarity' not in df.columns:
        if HAS_TEXTBLOB:
            df['Sentiment_Polarity'] = df['Translated_Review'].apply(
                lambda t: TextBlob(t).sentiment.polarity
            )
        else:
            df['Sentiment_Polarity'] = 0.0

    # Label sentiment if not present
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = df['Sentiment_Polarity'].apply(_label)

    df['Sentiment'] = df['Sentiment'].str.strip().str.capitalize()
    return df


def _label(polarity: float) -> str:
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    return 'Neutral'


# ──────────────────────────────────────────────
# TEXT HELPERS
# ──────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, remove stopwords."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


def top_words(texts: pd.Series, n: int = 20) -> pd.DataFrame:
    """Return DataFrame of top n words from a Series of review texts."""
    all_words = []
    for t in texts.dropna():
        all_words.extend(tokenize(str(t)))
    counts = Counter(all_words).most_common(n)
    return pd.DataFrame(counts, columns=['Word', 'Count'])


def wordcloud_text(texts: pd.Series) -> str:
    """Return single space-joined string of all meaningful words."""
    all_words = []
    for t in texts.dropna():
        all_words.extend(tokenize(str(t)))
    return ' '.join(all_words)


# ──────────────────────────────────────────────
# SUMMARY STATS
# ──────────────────────────────────────────────

def sentiment_summary(df: pd.DataFrame) -> dict:
    """Return count + pct for each sentiment class."""
    counts = df['Sentiment'].value_counts()
    total  = len(df)
    result = {}
    for label in ['Positive', 'Negative', 'Neutral']:
        c = int(counts.get(label, 0))
        result[label] = {'count': c, 'pct': round(c / total * 100, 1)}
    return result
