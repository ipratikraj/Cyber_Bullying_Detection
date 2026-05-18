"""
train_model.py
--------------
Trains a Logistic Regression model to detect cyberbullying in text.
Run this script once to generate the saved model files.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import os
import re

# ── 1. Sample dataset ────────────────────────────────────────────────────────
# In a real project you would load a CSV here, e.g.:
#   df = pd.read_csv("data/cyberbullying_dataset.csv")
# We include a balanced built-in dataset so the project works out of the box.

SAMPLES = [
    # Cyberbullying (label = 1)
    ("You are so ugly, nobody will ever like you", 1),
    ("I hate you so much, just disappear", 1),
    ("You're worthless and everyone knows it", 1),
    ("Kill yourself loser", 1),
    ("Nobody likes you, you're pathetic", 1),
    ("You're fat and disgusting", 1),
    ("I will find you and make your life hell", 1),
    ("You don't deserve to live", 1),
    ("Everyone hates you, why are you even here", 1),
    ("You're a complete idiot, go die", 1),
    ("Stop existing, you waste of space", 1),
    ("You're a freak and nobody wants you around", 1),
    ("I hope something bad happens to you", 1),
    ("You should be ashamed of yourself, loser", 1),
    ("Ugly pig, stay off the internet", 1),
    ("You're so dumb it's embarrassing", 1),
    ("Go back to where you came from, nobody wants you", 1),
    ("Your life is a joke, everyone laughs at you", 1),
    ("You're a failure in every way possible", 1),
    ("Stupid idiot, you ruin everything", 1),
    ("You make me sick just looking at you", 1),
    ("Trash human being, the world is better without you", 1),
    ("You're nothing but a burden to everyone", 1),
    ("No one will ever love someone as disgusting as you", 1),
    ("I bet even your parents hate you", 1),
    ("Such a disgusting creature shouldn't use the internet", 1),
    ("You're the reason people hate this place", 1),
    ("Reported you. Hope you get banned forever, garbage", 1),
    ("You're subhuman and don't belong here", 1),
    ("I will ruin your reputation everywhere", 1),

    # Not cyberbullying (label = 0)
    ("Have a wonderful day, hope you're doing well", 0),
    ("Great job on your project, really impressive work", 0),
    ("I love spending time with my friends and family", 0),
    ("The weather is so nice today, let's go for a walk", 0),
    ("Congratulations on your achievement, you earned it", 0),
    ("Thank you for your help, it meant a lot to me", 0),
    ("I really enjoyed reading your post", 0),
    ("You did an amazing job on that presentation", 0),
    ("Happy birthday! Hope your day is full of joy", 0),
    ("This is such a beautiful photo", 0),
    ("I think we should work together on this project", 0),
    ("Your opinion is really thoughtful and well reasoned", 0),
    ("Can't wait to see you at the event this weekend", 0),
    ("I disagree with your point, but I respect your view", 0),
    ("This movie was really interesting, what did you think", 0),
    ("Thank you for sharing your experience with us", 0),
    ("You have a real talent for writing", 0),
    ("Let's catch up over coffee sometime soon", 0),
    ("I appreciate your hard work and dedication", 0),
    ("The community would love to have your input", 0),
    ("That's a really creative idea, I hadn't thought of that", 0),
    ("Sending positive vibes your way today", 0),
    ("Your kindness makes a difference to everyone around you", 0),
    ("Well done on passing your exam", 0),
    ("I love how you always find the bright side", 0),
    ("This tutorial was so helpful, thanks for making it", 0),
    ("Keep going, you're doing better than you think", 0),
    ("What a lovely photo of your family", 0),
    ("You always have such thoughtful things to say", 0),
    ("Looking forward to collaborating with you", 0),
]

df = pd.DataFrame(SAMPLES, columns=["text", "label"])


# ── 2. Text cleaning ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()             # normalise spaces
    return text


df["clean"] = df["text"].apply(clean_text)


# ── 3. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


# ── 4. Feature extraction ────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)


# ── 5. Model training ────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
model.fit(X_train_vec, y_train)


# ── 6. Evaluation ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
acc    = accuracy_score(y_test, y_pred)

print("=" * 50)
print("  CYBERBULLYING DETECTION MODEL — RESULTS")
print("=" * 50)
print(f"\n  Accuracy : {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Not Bullying", "Cyberbullying"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("=" * 50)


# ── 7. Save model artefacts ──────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,      "model/cyberbullying_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n✅  Model saved to  model/cyberbullying_model.pkl")
print("✅  Vectorizer saved to  model/tfidf_vectorizer.pkl")
