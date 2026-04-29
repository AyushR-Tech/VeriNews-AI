# 📰 VeriNews AI

A **Flask web application** that classifies news as **real** or **fake** using a trained machine learning model.

---

## 🚀 Features

- 📰 Detects fake news from text input
- 🧠 Uses a pre-trained classification model
- 🌐 Simple web interface built with Flask
- 🎨 Responsive layout for desktop and mobile

---

## 📁 Project Structure

```
VeriNews-AI/
│
├── app.py
├── model.py
├── requirements.txt
├── tfidf_vectorizer.pkl
├── text_classification_model.pkl
│
├── data/
│   └── WELFake_Dataset.csv
│
├── static/
│   ├── style.css
│   └── Image.jpg
│
└── templates/
    └── index.html
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AyushR-Tech/VeriNews-AI.git
   cd VeriNews-AI
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are available**
   - `text_classification_model.pkl`
   - `tfidf_vectorizer.pkl`
   *(Both should be in the project root directory)*

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 Usage

1. Enter a news headline or article text.
2. Click **Predict**.
3. The app will display whether the news is likely **real** or **fake**.

---

## 📄 File Descriptions

- `app.py` – Flask app and web server logic
- `model.py` – Model training and prediction helper functions
- `requirements.txt` – Python dependencies
- `templates/index.html` – Web UI template
- `static/style.css` – App styling
- `tfidf_vectorizer.pkl` – Text vectorizer for preprocessing
- `text_classification_model.pkl` – Trained fake-news classifier

---

## 👨‍💻 Developed By

**Ayush Rewatkar**

---

## 📜 License

© 2026 Fake News Detection System. All rights reserved.
