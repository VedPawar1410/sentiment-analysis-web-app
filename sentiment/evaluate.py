"""Score the trained pipeline and write reports/metrics.json plus two charts.

Two test sets:
- tweets_test:   the 20% of Sentiment140 the model never trained on (same domain)
- amazon_unseen: Amazon product reviews (different domain - tests how well it generalises)

Usage (re-scores the saved model without retraining):
    python -m sentiment.evaluate
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # write image files, no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from sentiment import config, data
from sentiment.predict import load_model

EXAMPLES = [
    "I love this product, it is amazing!",
    "This is terrible, worst purchase ever",
    "not good at all",
    "not bad, actually pretty decent",
    "The battery life is great",
    "It was okay",
]

# Chart styling (light surface, recessive axes, one blue ramp for magnitude).
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"tweets_test": "#2a78d6", "amazon_unseen": "#eb6834"}
TITLES = {"tweets_test": "Tweets (unseen 20%)", "amazon_unseen": "Amazon reviews (unseen domain)"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#104281"])


def positive_probability(model, texts):
    positive_column = list(model.classes_).index(config.POSITIVE)
    return model.predict_proba(texts)[:, positive_column]


def score(labels, p_positive) -> dict:
    """Binary metrics at the 0.5 threshold, plus how often the app would say 'neutral'."""
    predicted = (p_positive >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[config.NEGATIVE, config.POSITIVE], zero_division=0
    )
    per_class = {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(["negative", "positive"])
    }
    in_band = (p_positive >= config.NEUTRAL_LOW) & (p_positive <= config.NEUTRAL_HIGH)
    return {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "roc_auc": round(float(roc_auc_score(labels, p_positive)), 4),
        **per_class,
        # rows = true [negative, positive], columns = predicted [negative, positive]
        "confusion_matrix": confusion_matrix(labels, predicted).tolist(),
        "neutral_share": round(float(in_band.mean()), 4),
    }


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_confusion_matrices(metrics: dict, path):
    """Row-normalised, so each row shows recall for that true class."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for ax, name in zip(axes, ["tweets_test", "amazon_unseen"]):
        cm = metrics[name]["confusion_matrix"]
        rows = [[v / sum(row) for v in row] for row in cm]
        ax.imshow(rows, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{rows[i][j]:.0%}\n({cm[i][j]:,})",
                    ha="center", va="center", fontsize=11,
                    color="#ffffff" if rows[i][j] > 0.5 else INK,
                )
        ax.set_xticks([0, 1], ["negative", "positive"])
        ax.set_yticks([0, 1], ["negative", "positive"])
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        ax.set_title(TITLES[name], color=INK, fontsize=12, loc="left")
        ax.tick_params(length=0, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Confusion matrix (% of each actual class)", color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_roc_curves(curves: dict, metrics: dict, path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Random guess")
    for name, (fpr, tpr) in curves.items():
        auc = metrics[name]["roc_auc"]
        ax.plot(fpr, tpr, color=SERIES[name], linewidth=2, label=f"{TITLES[name]} - AUC {auc:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", color=MUTED)
    ax.set_ylabel("True positive rate", color=MUTED)
    ax.set_title("ROC curve", color=INK, fontsize=13, loc="left")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(model, tweet_texts, tweet_labels, amazon) -> dict:
    """Score the model on both test sets and write metrics.json + charts to reports/."""
    test_sets = {
        "tweets_test": (tweet_texts, tweet_labels),
        "amazon_unseen": (amazon["text"], amazon["label"]),
    }
    metrics, curves = {}, {}
    for name, (texts, labels) in test_sets.items():
        p = positive_probability(model, texts)
        metrics[name] = score(labels, p)
        fpr, tpr, _ = roc_curve(labels, p)
        curves[name] = (fpr, tpr)

    metrics["examples"] = {
        text: round(float(p), 3) for text, p in zip(EXAMPLES, positive_probability(model, EXAMPLES))
    }

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrices(metrics, config.REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curves(curves, metrics, config.REPORTS_DIR / "roc_curve.png")

    for name in test_sets:
        m = metrics[name]
        print(
            f"{name:14s} n={m['n']:>7,}  accuracy={m['accuracy']:.3f}  auc={m['roc_auc']:.3f}  "
            f"recall neg={m['negative']['recall']:.3f} pos={m['positive']['recall']:.3f}"
        )
    print(f"Wrote {config.REPORTS_DIR / 'metrics.json'} and charts")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tweets", default=config.TWEETS_CSV, help="path to the Sentiment140 CSV")
    parser.add_argument("--amazon", default=config.AMAZON_CSV, help="path to the Amazon reviews CSV")
    args = parser.parse_args()

    _, X_test, _, y_test = data.split_tweets(data.load_tweets(args.tweets))
    run(load_model(), X_test, y_test, data.load_amazon(args.amazon))


if __name__ == "__main__":
    main()
