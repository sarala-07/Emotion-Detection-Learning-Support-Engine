from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Pre-trained BERT model
MODEL_NAME = "bert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6
)

print("✅ BERT Tokenizer Loaded Successfully!")
print("✅ BERT Model Loaded Successfully!")

print("\nModel Configuration:")
print(model.config)