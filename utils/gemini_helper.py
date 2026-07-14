import time
from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"

def get_learning_guidance(emotion, text):

    prompt = f"""
You are an AI Learning Assistant.

Emotion: {emotion}

Student Message:
{text}

Give:
1. Motivation
2. Study Advice
3. Learning Tip

Keep it short and encouraging.
"""

    for _ in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Attempt failed: {e}")
            time.sleep(5)
    return "Gemini service is busy. Please try again later."