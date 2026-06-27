import os
import sys
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# =====================================================
# Fix Windows UTF-8 Encoding
# =====================================================
os.environ["PYTHONUTF8"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =====================================================
# Download NLTK Resources (Runs only first time)
# =====================================================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# =====================================================
# Dataset Path
# =====================================================
DATASET = "youtube_bigdata.csv"

# =====================================================
# Load Dataset
# =====================================================
try:
    df = pd.read_csv(DATASET, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET, encoding="latin1")

df = df[['video_id', 'comment']].dropna()

print("="*60)
print("Dataset Loaded Successfully")
print("Rows :", len(df))
print("="*60)

# =====================================================
# NLP Objects
# =====================================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =====================================================
# Text Cleaning
# =====================================================
def clean_text(text):

    text = str(text).lower()

    # remove emoji
    text = text.encode("ascii", "ignore").decode("ascii")

    # remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    # remove stop words
    words = [w for w in words if w not in stop_words]

    # lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

df["clean_comment"] = df["comment"].apply(clean_text)

print("Text Cleaning Completed")

# =====================================================
# Sentiment Analysis
# =====================================================
def sentiment_score(text):
    return TextBlob(str(text)).sentiment.polarity

df["sentiment_score"] = df["comment"].apply(sentiment_score)

def sentiment_label(score):

    if score > 0.05:
        return 1

    elif score < -0.05:
        return -1

    else:
        return 0

df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)

print("Sentiment Analysis Completed")

# =====================================================
# Complaint Score
# =====================================================
complaint_words = [
    "accident",
    "crash",
    "danger",
    "fast",
    "speed",
    "traffic",
    "jam",
    "bad",
    "unsafe",
    "hurt",
    "road",
    "signal",
    "reckless"
]

def complaint_score(text):

    score = 0

    for word in complaint_words:

        if word in text:
            score += 1

    return score

df["complaint_score"] = df["clean_comment"].apply(complaint_score)

print("Complaint Score Completed")

# =====================================================
# TF-IDF
# =====================================================
tfidf = TfidfVectorizer(max_features=1000)

tfidf_matrix = tfidf.fit_transform(df["clean_comment"])

print("TF-IDF Completed")

# =====================================================
# SVD
# =====================================================
svd = TruncatedSVD(
    n_components=2,
    random_state=42
)

svd_result = svd.fit_transform(tfidf_matrix)

df["svd_1"] = svd_result[:,0]
df["svd_2"] = svd_result[:,1]

print("SVD Completed")

# =====================================================
# Severity Score
# =====================================================
df["severity_score"] = (
    (df["complaint_score"] * 2)
    - df["sentiment_score"]
)

print("Severity Score Calculated")

# =====================================================
# Final Output
# =====================================================
final_df = df[
    [
        "video_id",
        "comment",
        "clean_comment",
        "sentiment_score",
        "sentiment_label",
        "complaint_score",
        "severity_score",
        "svd_1",
        "svd_2"
    ]
]

print("="*60)
print("First Five Rows")
print("="*60)

print(final_df.head().to_string())

print("="*60)

# =====================================================
# Save CSV
# =====================================================
OUTPUT = "output_features.csv"

final_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("Output Saved Successfully")
print("File :", OUTPUT)

print("Total Rows :", len(final_df))
print("Total Columns :", len(final_df.columns))

print("="*60)

print("\nFinished Successfully.")

print("\nOverall Public Opinion")

counts = df["sentiment_label"].value_counts()

print(counts)

print("\nFinal Opinion:",
      "Positive" if counts.get(1,0) > counts.get(-1,0)
      else "Negative" if counts.get(-1,0) > counts.get(1,0)
      else "Neutral")