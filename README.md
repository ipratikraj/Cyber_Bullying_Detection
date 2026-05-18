# 🛡️ CyberShield — AI Cyberbullying Detection System

An AI-powered web application that detects cyberbullying in text messages using **Machine Learning** and **Natural Language Processing (NLP)**.

---

## 📸 Preview

> A simple web-based interface for detecting cyberbullying text using AI techniques.

---

## ✨ Features

- 🔍 **Real-time detection** — paste any message and get an instant result
- 📊 **Confidence score** — shows how certain the model is (e.g. 94.2%)
- ✅ / 🚨 **Clear verdict** — "Safe" or "Cyberbullying Detected"
- 💡 **Helpful tip** — guidance on what to do next
- 🌐 **Web interface** — no command-line needed after setup

---

## 🧠 How It Works

```
User Input → Text Cleaning → TF-IDF Vectorizer → Logistic Regression → Prediction
```

1. **Text Cleaning** — Converts to lowercase, removes URLs and punctuation
2. **TF-IDF Vectorizer** — Converts words into numerical feature vectors
3. **Logistic Regression** — Classifies the text as bullying or not
4. **Result** — Returns label + confidence score to the web page

---

## 🛠️ Tech Stack

| Layer       | Technology                     |
|-------------|--------------------------------|
| Backend     | Python 3, Flask                |
| ML / NLP    | scikit-learn, TF-IDF, joblib   |
| Frontend    | HTML, CSS, Vanilla JavaScript  |
| Model       | Logistic Regression            |

---

## 📁 Project Structure

```
cyberbullying-detection/
│
├── app.py                  ← Flask web application (main server)
├── train_model.py          ← Script to train and save the ML model
├── requirements.txt        ← Python dependencies
├── .gitignore
├── README.md
│
├── model/                  ← Saved model files (generated after training)
│   ├── cyberbullying_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── templates/              ← HTML pages
    ├── index.html          ← Main detection page
    └── about.html          ← About / how it works page
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ipratikraj/Cyber_Bullying_Detection.git
cd Cyber_Bullying_Detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

This generates the `.pkl` model files inside the `model/` folder.

```bash
python train_model.py
```

You should see output like:
```
  Accuracy : 91.67%
✅  Model saved to  model/cyberbullying_model.pkl
✅  Vectorizer saved to  model/tfidf_vectorizer.pkl
```

### 5. Run the Web App

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 📊 Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 91.67%  |
| Precision | 93%     |
| Recall    | 92%     |
| F1-Score  | 92%     |


---

## 👨‍💻 Author

Developed by Pratik Raj

MCA Student — Cochin University of Science and Technology (CUSAT)
