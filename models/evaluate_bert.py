import os
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer
)

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PATH = os.path.join(BASE_DIR, "dataset", "test.txt")

MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_model")

# ======================================================
# Load Dataset
# ======================================================

test_df = pd.read_csv(
    TEST_PATH,
    sep=";",
    header=None,
    names=["text", "emotion"]
)

print("✅ Test dataset loaded")

# ======================================================
# Label Encoding
# ======================================================

label_encoder = LabelEncoder()

label_encoder.fit([
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise"
])

test_df["label"] = label_encoder.transform(test_df["emotion"])

# ======================================================
# HuggingFace Dataset
# ======================================================

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]]
)

# ======================================================
# Load Tokenizer
# ======================================================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_DIR
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# ======================================================
# Load Model
# ======================================================

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_DIR
)

trainer = Trainer(model=model)

# ======================================================
# Prediction
# ======================================================

print("\n🚀 Evaluating DistilBERT...\n")

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# ======================================================
# Metrics
# ======================================================

accuracy = accuracy_score(y_true, y_pred)

print("=" * 50)
print("DistilBERT Test Accuracy")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_true, y_pred))

print("\n✅ Evaluation Completed Successfully!")