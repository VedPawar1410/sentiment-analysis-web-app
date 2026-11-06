"""Every path and setting in one place, so training and the web app can't disagree."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Datasets live in data/ (git-ignored, see data/README.md for download links).
DATA_DIR = ROOT / "data"
TWEETS_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"
AMAZON_CSV = DATA_DIR / "7817_1.csv"

# The trained pipeline (vectorizer + classifier saved together as one file).
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.joblib"

# Metrics and charts written by evaluation, referenced from the README.
REPORTS_DIR = ROOT / "reports"

# Labels used everywhere in this codebase.
# Sentiment140 stores positive as 4 - we convert it to 1 as soon as the data is loaded.
NEGATIVE, POSITIVE = 0, 1
LABEL_NAMES = {NEGATIVE: "negative", POSITIVE: "positive"}

# Train/test split.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# If the model's positive probability lands in this band, it isn't sure either way,
# so the app shows "neutral" instead of forcing a side.
NEUTRAL_LOW, NEUTRAL_HIGH = 0.4, 0.6
