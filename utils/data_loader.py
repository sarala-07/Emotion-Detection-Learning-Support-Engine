import os
import pandas as pd

# Get absolute path of the project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def load_dataset():
    train = pd.read_csv(
        os.path.join(DATASET_DIR, "train.txt"),
        sep=";",
        names=["text", "emotion"]
    )

    val = pd.read_csv(
        os.path.join(DATASET_DIR, "val.txt"),
        sep=";",
        names=["text", "emotion"]
    )

    test = pd.read_csv(
        os.path.join(DATASET_DIR, "test.txt"),
        sep=";",
        names=["text", "emotion"]
    )

    return train, val, test

if __name__ == "__main__":
    train_df, val_df, test_df = load_dataset()

    print("=" * 50)
    print("Training Dataset")
    print("=" * 50)
    print(train_df.head())

    print("\nTraining Shape:", train_df.shape)

    print("\nValidation Shape:", val_df.shape)
    print("Test Shape:", test_df.shape)

    print("\nEmotion Distribution:")
    print(train_df["emotion"].value_counts())

    print("\nMissing Values:")
    print(train_df.isnull().sum())