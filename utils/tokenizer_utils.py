import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_loader import load_dataset
from preprocessing import clean_text

# Load datasets
train_df, val_df, test_df = load_dataset()

# Clean text
train_texts = train_df["text"].apply(clean_text)
val_texts = val_df["text"].apply(clean_text)
test_texts = test_df["text"].apply(clean_text)

# Create tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# Fit tokenizer
tokenizer.fit_on_texts(train_texts)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
MAX_LENGTH = 100

X_train = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding="post")
X_val = pad_sequences(val_sequences, maxlen=MAX_LENGTH, padding="post")
X_test = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding="post")

# Save tokenizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved successfully!")

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

print("\nVocabulary Size:", len(tokenizer.word_index))
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)

print("✅ Tokenized datasets saved successfully!")