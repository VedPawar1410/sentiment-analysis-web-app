# Sentiment Analysis Web Application

This repository contains a web application that performs sentiment analysis on text input, predicting whether the sentiment is positive or negative. The application is built using Flask (Python), and the machine learning model used is a Logistic Regression model trained on sentiment-labeled data.

## Features
* A simple web interface for entering text (such as a review or a sentence).
* Sentiment prediction displayed instantly upon form submission.
* Deployed with Flask, styled with HTML, CSS, and basic JavaScript.
* Sentiment prediction is based on a pre-trained model using Scikit-learn.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/VedPawar1410/sentiment-analysis-app.git](https://github.com/VedPawar1410/sentiment-analysis-app.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd sentiment-analysis-app
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Flask app:
    ```bash
    python app.py
    ```
5.  Open a browser and navigate to `http://127.0.0.1:5000/` to use the application.

## File Structure

* `app.py`: The main backend script, handling the sentiment prediction.
* `templates/`: Contains the HTML templates (`index.html`, `result.html`).
* `static/`: Contains CSS files for styling.
* `model.pkl`: Pre-trained logistic regression model for sentiment prediction.
* `vectorizer.pkl`: The TF-IDF vectorizer used for text transformation.

## Usage

1.  Launch the Flask server.
2.  Open the web application in your browser.
3.  Enter a sentence or review in the input box.
4.  Click on the 'Predict' button to see whether the sentiment is positive or negative.

## Future Improvements

* Option to upload bulk text data for batch sentiment analysis.
* More detailed analysis, including neutral sentiment detection.
* Improved UI/UX for better user interaction.