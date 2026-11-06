from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer from the .pkl files
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer1.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a function to predict sentiment
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])  # Vectorize the input text
    prediction = model.predict(text_vector)[0]  # Predict using the trained model
    # Sentiment140 labels: 0 = negative, 4 = positive (NOT 1)
    return 'positive' if prediction == 4 else 'negative'

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle form submission and predict sentiment
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form (from the textarea with name 'user_input')
    user_input = request.form['user_input']  # Ensure the name matches the HTML form
    result = predict_sentiment(user_input)   # Predict sentiment based on input
    return render_template('result.html', prediction=result)

# Run the app
#if __name__ == '__main__':
    #app.run(debug=True)
