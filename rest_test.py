import requests
from config import GEMINI_API_KEY

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Say hello."
                }
            ]
        }
    ]
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.text)