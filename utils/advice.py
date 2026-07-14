def get_learning_advice(emotion):

    advice = {

        "joy": {
            "motivation": "Great! Keep this positive energy going.",
            "study": "This is a good time to tackle challenging topics.",
            "tip": "Revise what you've learned today."
        },

        "sadness": {
            "motivation": "Every difficult day passes. Keep moving forward.",
            "study": "Take a short break before continuing your studies.",
            "tip": "Study for 25 minutes, then rest for 5 minutes."
        },

        "anger": {
            "motivation": "Stay calm. You can overcome this.",
            "study": "Avoid studying while frustrated. Relax first.",
            "tip": "Take deep breaths and then begin with easy topics."
        },

        "fear": {
            "motivation": "Believe in yourself. You're more prepared than you think.",
            "study": "Break large chapters into smaller sections.",
            "tip": "Practice previous exam questions."
        },

        "love": {
            "motivation": "Positive emotions help learning.",
            "study": "Stay focused while enjoying the moment.",
            "tip": "Teach someone what you learned."
        },

        "surprise": {
            "motivation": "Unexpected things can become opportunities.",
            "study": "Review today's new concepts carefully.",
            "tip": "Write down important points immediately."
        }

    }

    return advice.get(
        emotion,
        {
            "motivation": "Keep learning every day.",
            "study": "Stay focused.",
            "tip": "Practice regularly."
        }
    )