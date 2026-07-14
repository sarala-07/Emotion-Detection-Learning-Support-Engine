from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

print("Available models:\n")

for model in client.models.list():
    print(model.name)
    if hasattr(model, "supported_actions"):
        print("  Supported actions:", model.supported_actions)