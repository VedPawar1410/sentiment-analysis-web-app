"""Checks against the real trained model in models/."""

import pytest

from sentiment import config
from sentiment.predict import load_model, predict_sentiment


def test_model_uses_0_and_1_labels():
    # Guards against the original bug: Sentiment140's raw label for positive is 4, not 1.
    assert list(load_model().classes_) == [config.NEGATIVE, config.POSITIVE]


@pytest.mark.parametrize(
    "text",
    ["I love this product, it is amazing!", "Best day ever, so happy", "not bad, actually pretty decent"],
)
def test_positive_text(text):
    assert predict_sentiment(text)[0] == "positive"


@pytest.mark.parametrize(
    "text",
    ["This is terrible, worst purchase ever", "I hate this", "not good at all"],
)
def test_negative_text(text):
    assert predict_sentiment(text)[0] == "negative"


def test_probability_is_between_0_and_1():
    _, p = predict_sentiment("hello")
    assert 0.0 <= p <= 1.0
