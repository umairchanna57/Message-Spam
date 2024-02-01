from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset from CSV file
df = pd.read_csv('Roman Urdu DataSet.csv')

# Handle NaN values by replacing them with an empty string
df['Message'].fillna('', inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Initialize Naive Bayes model
nb_classifier = MultinomialNB()

# Train the Naive Bayes model
nb_classifier.fit(X_train_tfidf, y_train)

# Flask app initialization
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input message from the form
        message = request.form['message']

        # Handle NaN values in the input message
        if pd.isna(message):
            message = ''

        # Vectorize the input message using the trained TF-IDF Vectorizer
        message_tfidf = tfidf_vectorizer.transform([message])

        # Make a prediction using the trained Naive Bayes model
        prediction = nb_classifier.predict(message_tfidf)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
