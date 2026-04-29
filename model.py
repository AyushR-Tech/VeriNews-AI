import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv('data/WELFake_Dataset.csv')

# Use only headline (IMPORTANT)
df = df[['title', 'label']]

# Drop missing values
df.dropna(inplace=True)

# -------------------------------
# Step 2: Text Cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['title'].apply(clean_text)
df['processed'] = df['cleaned'].apply(preprocess_text)

# -------------------------------
# Step 3: Features & Labels
# -------------------------------
X = df['processed']
y = df['label']

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 5: TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# Step 6: Random Forest Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# -------------------------------
# Step 7: Evaluation
# -------------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Save Model
# -------------------------------
joblib.dump(model, 'rf_fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# -------------------------------
# Step 9: Prediction Function
# -------------------------------
def predict_news(headline):
    model = joblib.load('rf_fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    cleaned = clean_text(headline)
    processed = preprocess_text(cleaned)

    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]

    return "Fake News" if prediction == 1 else "Real News"

# -------------------------------
# Step 10: User Input
# -------------------------------
while True:
    text = input("\nEnter news headline (or 'exit'): ")
    if text.lower() == 'exit':
        break
    print("Prediction:", predict_news(text))