# Data

The CSV files are **not committed** (the tweet file is 228 MB). Download them and put them in this folder:

| File | What it is | Used for |
|---|---|---|
| `training.1600000.processed.noemoticon.csv` | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140): 1.6M tweets from 2009, half negative, half positive. Labelled automatically from emoticons (`:)` = positive, `:(` = negative), and the emoticons were then removed. | Training (80%) and testing (20%) |
| `7817_1.csv` | ~1,600 Amazon product reviews (Kindle, Fire TV, Echo...) with 1-5 star ratings, in Datafiniti's product-review format. A copy is mirrored on [Hugging Face](https://huggingface.co/d-e-e-k-11/amazon_product/blob/main/7817_1.csv). | Testing only, to see how the model handles text that isn't tweets |

## Sentiment140 format

No header row. 6 columns:

| # | Column | Example |
|---|---|---|
| 0 | target | `0` = negative, `4` = positive (**not 1** - the code converts 4 to 1 on load) |
| 1 | id | `1467810369` |
| 2 | date | `Mon Apr 06 22:19:45 PDT 2009` |
| 3 | flag | `NO_QUERY` |
| 4 | user | `some_username` |
| 5 | text | the tweet |

Read it with `encoding="latin-1"`; it isn't valid UTF-8.

## Amazon reviews

Only `reviews.text` and `reviews.rating` are used. 4-5 stars = positive, 1-2 stars = negative, 3 stars dropped.
That leaves 1,053 reviews, and only 76 of them are negative, so look at per-class recall, not just accuracy.

You can also keep the files elsewhere and pass their paths:

```bash
python -m sentiment.train --tweets /path/to/tweets.csv --amazon /path/to/7817_1.csv
```
