from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model
model = joblib.load('rf_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Safe NLTK loading
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def preprocess_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No input provided"})

        user_input = data["text"].strip()

        if not user_input:
            return jsonify({"error": "Empty input"})

        cleaned = clean_text(user_input)
        processed = preprocess_text(cleaned)

        vector = vectorizer.transform([processed])

        if vector.shape[1] == 0:
            return jsonify({"error": "Invalid input"})

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        result = "Fake News" if prediction == 1 else "Real News"

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)