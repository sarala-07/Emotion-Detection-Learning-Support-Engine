import csv
import os
from datetime import datetime

LOG_FILE = "logs/emotion_history.csv"


def save_prediction(text, emotion, confidence):

    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Student Text",
                "Emotion",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,
            emotion,
            round(confidence, 2)
        ])