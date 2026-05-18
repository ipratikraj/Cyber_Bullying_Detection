"""
app.py
------
Flask web application for the Cyberbullying Detection System.

Run with:
    python app.py
Then open  http://127.0.0.1:5000  in your browser.
"""

import os
import re
import json
import joblib
from flask import Flask, render_template, request, jsonify

# ── Initialise Flask ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model artefacts ──────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("model", "cyberbullying_model.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ── Text pre-processing (must match train_model.py) ──────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Helper ────────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    """Return prediction label, confidence, and a short message."""
    cleaned   = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    label      = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    confidence = round(float(proba[label]) * 100, 2)

    if label == 1:
        result  = "Cyberbullying Detected"
        verdict = "danger"
        tip     = (
            "This message contains harmful language. "
            "Please consider reporting it and supporting the affected individual."
        )
    else:
        result  = "No Cyberbullying Detected"
        verdict = "safe"
        tip     = "This message appears to be safe and respectful."

    return {
        "result":     result,
        "verdict":    verdict,
        "confidence": confidence,
        "tip":        tip,
        "input":      text,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text is too long (max 1000 characters)."}), 400

    return jsonify(predict(text))


@app.route("/about")
def about():
    return render_template("about.html")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
