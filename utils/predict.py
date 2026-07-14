import os
import torch
import numpy as np

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
# ======================================================
# Project Paths
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_model"
)

print("Loading model from:", MODEL_DIR)
# ======================================================
# Load Tokenizer and Model
# ======================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

print("✅ DistilBERT model loaded successfully!")
# ======================================================
# Emotion Labels
# ======================================================

LABELS = [
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise"
]
# ======================================================
# Emotion Prediction Function
# ======================================================

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    emotion = LABELS[prediction.item()]

    confidence = confidence.item() * 100

    return emotion, confidence
# ======================================================
# Test Prediction
# ======================================================

if __name__ == "__main__":

    sample_texts = [
        "I am feeling very happy today!",
        "I am extremely sad and disappointed.",
        "I am really angry with my friend.",
        "I am scared about tomorrow's exam.",
        "I love spending time with my family.",
        "Wow! I didn't expect this at all."
    ]

    print("\n========== Emotion Prediction ==========\n")

    for text in sample_texts:

        emotion, confidence = predict_emotion(text)

        print(f"Text       : {text}")
        print(f"Emotion    : {emotion}")
        print(f"Confidence : {confidence:.2f}%")
        print("-" * 50)