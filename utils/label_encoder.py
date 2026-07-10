import os
import joblib
from sklearn.preprocessing import LabelEncoder

from data_loader import load_dataset

# Load datasets
train_df, val_df, test_df = load_dataset()

# Create encoder
label_encoder = LabelEncoder()

# Fit on training labels
train_df["label"] = label_encoder.fit_transform(train_df["emotion"])
val_df["label"] = label_encoder.transform(val_df["emotion"])
test_df["label"] = label_encoder.transform(test_df["emotion"])

# Display mapping
print("\nEmotion Mapping:")
for emotion, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{emotion} --> {label}")

# Save encoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

print("\n✅ Label encoder saved successfully!")
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), train_df["label"].values)
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), val_df["label"].values)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), test_df["label"].values)

print("✅ Labels saved successfully!")