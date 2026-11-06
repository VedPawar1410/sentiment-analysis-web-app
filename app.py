"""Flask web app: a form that sends text to the trained sentiment pipeline.

Local:      python app.py            (then open http://127.0.0.1:5000)
Production: gunicorn app:app         (what Render runs)
"""

from flask import Flask, redirect, render_template, request, url_for

from sentiment.predict import load_model, predict_sentiment

app = Flask(__name__)

# Load the model once at startup, so the first visitor doesn't wait for it.
load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("user_input", "").strip()
    if not user_input:
        return redirect(url_for("index"))
    label, _ = predict_sentiment(user_input)  # 'positive', 'negative' or 'neutral'
    return render_template("result.html", prediction=label)


# Only runs with `python app.py`. gunicorn imports `app` and never executes this block,
# so it's safe to keep for local development.
if __name__ == "__main__":
    app.run(debug=True)
