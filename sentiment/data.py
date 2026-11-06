"""Load the datasets and normalise labels to 0 = negative, 1 = positive."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment import config


def load_tweets(path: Path = config.TWEETS_CSV) -> pd.DataFrame:
    """Sentiment140: 1.6M tweets, no header row, label 0 = negative, 4 = positive."""
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=None,
        usecols=[0, 5],
        names=["target", "text"],
    )
    df["label"] = (df["target"] == 4).astype(int)
    return df[["text", "label"]]


def split_tweets(df: pd.DataFrame):
    """Fixed, stratified 80/20 split so training and evaluation always see the same test set."""
    return train_test_split(
        df["text"],
        df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )


def load_amazon(path: Path = config.AMAZON_CSV) -> pd.DataFrame:
    """Amazon product reviews, used ONLY as an unseen, out-of-domain test set.

    4-5 stars = positive, 1-2 stars = negative, 3 stars dropped (ambiguous).
    """
    df = pd.read_csv(path, usecols=["reviews.text", "reviews.rating"])
    df = df.dropna()
    df = df[df["reviews.rating"] != 3]
    return pd.DataFrame(
        {
            "text": df["reviews.text"],
            "label": (df["reviews.rating"] >= 4).astype(int),
        }
    )
