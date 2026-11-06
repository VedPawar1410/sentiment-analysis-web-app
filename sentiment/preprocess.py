"""Text cleaning shared by training and prediction.

This is the ONLY cleaning function. The vectorizer calls it inside the saved pipeline,
so the web app automatically cleans text exactly the way training did.

Deliberately NOT done (each one hurt sentiment in the original notebook):
- stopword removal: NLTK's list contains "not", "no", "nor" - removing them turns
  "not good" into "good".
- stemming: unnecessary for TF-IDF at this data size, and it forced the app to
  reproduce the exact same stemming at prediction time.
"""

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NEGATION_RE = re.compile(r"n't\b")


def clean_text(text: str) -> str:
    text = html.unescape(str(text))   # "&amp;" -> "&", "&quot;" -> '"'
    text = text.replace("’", "'").lower()  # curly apostrophe -> straight
    text = URL_RE.sub(" url ", text)          # every link looks the same
    text = MENTION_RE.sub(" user ", text)     # every @username looks the same
    text = NEGATION_RE.sub(" not", text)      # "isn't" -> "is not", so "not good" is learnable
    return text
