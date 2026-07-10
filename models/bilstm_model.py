import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# Load Processed Data
# -----------------------------
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))

y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# -----------------------------
# Load Model
# -----------------------------
model = load_model(os.path.join(MODELS_DIR, "bilstm_model.keras"))

print("\nModel Loaded Successfully!")

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, "best_bilstm.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# Save Final Model
# -----------------------------
model.save(os.path.join(MODELS_DIR, "bilstm_model.keras"))

print("\n🎉 BiLSTM Training Completed Successfully!")