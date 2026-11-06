from sentiment.preprocess import clean_text


def test_keeps_negation_words():
    # The original notebook removed stopwords, which deleted "not".
    assert "not" in clean_text("This is not good").split()


def test_expands_contractions_into_not():
    assert clean_text("It isn't good") == "it is not good"


def test_replaces_urls_and_mentions():
    cleaned = clean_text("@bob look at https://example.com/x")
    assert "bob" not in cleaned and "example" not in cleaned
    assert "user" in cleaned.split() and "url" in cleaned.split()


def test_unescapes_html():
    assert "&amp;" not in clean_text("fish &amp; chips")
