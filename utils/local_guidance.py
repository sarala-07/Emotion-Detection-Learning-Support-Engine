def get_learning_guidance(emotion):
    guidance = {
        "joy": """
🎉 Motivation:
Keep up the great work!

📚 Study Advice:
This is a perfect time to learn difficult concepts.

💡 Tip:
Take short notes while studying.
""",

        "sadness": """
❤️ Motivation:
Every difficult day passes. Keep believing in yourself.

📚 Study Advice:
Study one topic at a time without pressure.

💡 Tip:
Take a 10-minute break after every hour.
""",

        "fear": """
💪 Motivation:
You are stronger than your fears.

📚 Study Advice:
Practice previous questions to build confidence.

💡 Tip:
Break large tasks into smaller goals.
""",

        "anger": """
😊 Motivation:
Stay calm. Progress comes with patience.

📚 Study Advice:
Take a short walk before studying again.

💡 Tip:
Focus on understanding rather than memorizing.
""",

        "love": """
🌸 Motivation:
Keep spreading positivity.

📚 Study Advice:
Study with friends or discuss concepts.

💡 Tip:
Teach someone what you've learned.
""",

        "surprise": """
✨ Motivation:
Stay curious.

📚 Study Advice:
Explore new concepts while you're interested.

💡 Tip:
Write down anything surprising you discover.
"""
    }

    return guidance.get(
        emotion.lower(),
        """
😊 Motivation:
Believe in yourself.

📚 Study Advice:
Stay consistent.

💡 Tip:
Practice every day.
"""
    )
