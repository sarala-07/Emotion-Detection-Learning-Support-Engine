import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_bilstm.keras")

DATA_PATH = os.path.join(BASE_DIR, "processed_data")

# -----------------------------
# Load Model
# -----------------------------
model = load_model(MODEL_PATH)

print("✅ Model Loaded Successfully!")

# -----------------------------
# Load Test Data
# -----------------------------
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("==============================")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))