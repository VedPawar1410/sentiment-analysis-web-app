"""Train the sentiment pipeline on Sentiment140, save it, then evaluate it.

Usage:
    python -m sentiment.train
    python -m sentiment.train --tweets path/to/tweets.csv --amazon path/to/7817_1.csv

Takes about a minute on a laptop and writes:
    models/sentiment_pipeline.joblib
    reports/metrics.json, reports/confusion_matrix.png, reports/roc_curve.png
"""

import argparse
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment import config, data, evaluate
from sentiment.preprocess import clean_text


def build_pipeline() -> Pipeline:
    """Cleaning -> TF-IDF on words and word pairs -> Logistic Regression, as ONE object.

    Saving the whole pipeline (instead of a separate vectorizer.pkl and model.pkl)
    means the web app can't apply different cleaning or a different vectorizer by mistake.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,  # our cleaning runs inside the pipeline
                    ngram_range=(1, 2),       # words + word pairs: "not good" becomes one feature
                    min_df=3,                 # ignore terms seen in fewer than 3 tweets (typos, noise)
                    max_features=500_000,     # keep the 500k most frequent terms
                    sublinear_tf=True,        # "good good good" isn't 3x as positive as "good"
                ),
            ),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tweets", default=config.TWEETS_CSV, help="path to the Sentiment140 CSV")
    parser.add_argument("--amazon", default=config.AMAZON_CSV, help="path to the Amazon reviews CSV")
    args = parser.parse_args()

    print("Loading tweets...")
    tweets = data.load_tweets(args.tweets)
    X_train, X_test, y_train, y_test = data.split_tweets(tweets)
    print(f"Train: {len(X_train):,} tweets   Test: {len(X_test):,} tweets")

    print("Training...")
    start = time.time()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"Trained in {time.time() - start:.0f}s, vocabulary = {vocab_size:,} terms")

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH, compress=3)
    size_mb = config.MODEL_PATH.stat().st_size / 1e6
    print(f"Saved {config.MODEL_PATH.relative_to(config.ROOT)} ({size_mb:.1f} MB)")

    print("Evaluating...")
    evaluate.run(pipeline, X_test, y_test, data.load_amazon(args.amazon))


if __name__ == "__main__":
    main()
