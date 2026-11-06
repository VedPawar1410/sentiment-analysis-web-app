"""Load the trained pipeline and turn text into a sentiment label.

Usage from the command line:
    python -m sentiment.predict "I love this product"
"""

import sys
from functools import lru_cache

import joblib

from sentiment import config


@lru_cache(maxsize=1)
def load_model():
    """Load the pipeline once and reuse it for every request."""
    return joblib.load(config.MODEL_PATH)


def predict_sentiment(text: str) -> tuple[str, float]:
    """Return (label, positive_probability). label is 'positive', 'negative' or 'neutral'."""
    model = load_model()
    positive_column = list(model.classes_).index(config.POSITIVE)
    p_positive = float(model.predict_proba([text])[0, positive_column])

    if p_positive > config.NEUTRAL_HIGH:
        label = "positive"
    elif p_positive < config.NEUTRAL_LOW:
        label = "negative"
    else:
        label = "neutral"
    return label, p_positive


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Text: ")
    label, p = predict_sentiment(text)
    print(f"{label}  (positive probability = {p:.2f})")
