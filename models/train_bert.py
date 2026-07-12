import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.txt")
VAL_PATH = os.path.join(BASE_DIR, "dataset", "val.txt")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test.txt")

MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_model")

# ======================================================
# Load Dataset
# ======================================================

train_df = pd.read_csv(
    TRAIN_PATH,
    sep=";",
    header=None,
    names=["text", "emotion"]
)

val_df = pd.read_csv(
    VAL_PATH,
    sep=";",
    header=None,
    names=["text", "emotion"]
)

test_df = pd.read_csv(
    TEST_PATH,
    sep=";",
    header=None,
    names=["text", "emotion"]
)

print("✅ Dataset Loaded")

# ======================================================
# Encode Labels
# ======================================================

label_encoder = LabelEncoder()

train_df["label"] = label_encoder.fit_transform(train_df["emotion"])
val_df["label"] = label_encoder.transform(val_df["emotion"])
test_df["label"] = label_encoder.transform(test_df["emotion"])

print("Classes:")
print(label_encoder.classes_)

# ======================================================
# HuggingFace Dataset
# ======================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]]
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]]
)

# ======================================================
# Tokenizer
# ======================================================

MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_NAME
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

columns = [
    "input_ids",
    "attention_mask",
    "label"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

val_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)

print("✅ Tokenization Complete")

# ======================================================
# Load Model
# ======================================================

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return accuracy.compute(
        predictions=predictions,
        references=labels
    )
# ======================================================
# Training Arguments
# ======================================================

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    overwrite_output_dir=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",
    greater_is_better=True,

    report_to="none"
)

# ======================================================
# Trainer
# ======================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n🚀 Starting DistilBERT Training...\n")

trainer.train()

print("\n✅ Training Completed!")

# ======================================================
# Evaluation
# ======================================================

results = trainer.evaluate(test_dataset)

print("\n==============================")
print("Test Results")
print("==============================")

for key, value in results.items():
    print(f"{key}: {value}")

# ======================================================
# Save Model
# ======================================================

os.makedirs(MODEL_DIR, exist_ok=True)

trainer.save_model(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print("\n✅ DistilBERT Model Saved Successfully!")
print(f"Location: {MODEL_DIR}")

print("\n🎉 DistilBERT Training Pipeline Completed!")