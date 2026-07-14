from utils.gemini_helper import get_learning_guidance

emotion = "sadness"

text = "I am feeling stressed because of my exams."

response = get_learning_guidance(emotion, text)

print("\n===== Gemini Response =====\n")
print(response)