import os
import streamlit as st

# ======================================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# ======================================================

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)

# ======================================================
# IMPORTS
# ======================================================

from utils.predict import predict_emotion
from utils.advice import get_learning_advice
from utils.logger import save_prediction
from utils.dashboard import (
    load_history,
    emotion_bar_chart,
    emotion_pie_chart,
    confidence_chart
)

# ======================================================
# LOAD CSS
# ======================================================

def load_css():
    css_path = os.path.join("style", "style.css")

    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🤖 AI Learning Assistant")
with st.expander("ℹ️ About This Project"):

    st.write("""
### Emotion Detection Learning Support Engine

This AI-powered application detects a student's emotion from text
using a trained DistilBERT model.

### Features

- 🤖 Emotion Detection
- 📚 Personalized Learning Advice
- 📊 Analytics Dashboard
- 📜 Prediction History
- 📄 CSV Report Download

### Technologies Used

- Python
- Streamlit
- DistilBERT
- Hugging Face Transformers
- Plotly
- Pandas
- TensorFlow
""")
st.sidebar.markdown("---")

st.sidebar.info("""
### Features

✅ Emotion Detection

✅ Personalized Learning Support

✅ Prediction History

✅ Analytics Dashboard

✅ CSV Report Download
""")

st.sidebar.markdown("---")

st.sidebar.success("Model : DistilBERT")
st.sidebar.success("Framework : Streamlit")
st.sidebar.success("Analytics : Plotly")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Emotion Detection Learning Support Engine"
)

# ======================================================
# EMOJIS
# ======================================================

EMOJI = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# ======================================================
# TITLE
# ======================================================

st.title("🎓 Emotion Detection Learning Support Engine")

st.markdown("""
This application analyzes a student's emotions using a trained **DistilBERT**
model and provides personalized learning guidance.

Enter how you feel below and click **Predict Emotion**.
""")

# ======================================================
# USER INPUT
# ======================================================

user_text = st.text_area(
    "📝 Describe how you are feeling today",
    height=170,
    placeholder="Example: I am feeling stressed because of my exams..."
)

predict_button = st.button("🔍 Predict Emotion")

# ======================================================
# PREDICTION
# ======================================================

if predict_button:

    if user_text.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:

        emotion, confidence = predict_emotion(user_text)

        save_prediction(
            user_text,
            emotion,
            confidence
        )

        advice = get_learning_advice(emotion)

        st.success("✅ Emotion detected successfully!")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🎯 Prediction Result")

            st.markdown(
                f"## {EMOJI.get(emotion,'🤖')} {emotion.capitalize()}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

            if confidence >= 90:
                st.success("Excellent prediction confidence!")

            elif confidence >= 75:
                st.info("Good prediction confidence.")

            else:
                st.warning(
                    "Prediction confidence is relatively low. Try writing a longer sentence."
                )

        with col2:

            st.subheader("📚 Personalized Learning Support")

            st.success(
                f"💪 Motivation\n\n{advice['motivation']}"
            )

            st.info(
                f"📖 Study Advice\n\n{advice['study']}"
            )

            st.warning(
                f"💡 Learning Tip\n\n{advice['tip']}"
            )

# ======================================================
# ANALYTICS DASHBOARD
# ======================================================

st.divider()

st.header("📊 Analytics Dashboard")

history = load_history()

if not history.empty:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Predictions",
            len(history)
        )

    with c2:
        st.metric(
            "Most Common Emotion",
            history["Emotion"].mode()[0]
        )

    with c3:
        st.metric(
            "Average Confidence",
            f"{history['Confidence'].mean():.2f}%"
        )

    chart1, chart2 = st.columns(2)

    with chart1:
        st.plotly_chart(
            emotion_bar_chart(history),
            use_container_width=True
        )

    with chart2:
        st.plotly_chart(
            emotion_pie_chart(history),
            use_container_width=True
        )

    st.plotly_chart(
        confidence_chart(history),
        use_container_width=True
    )

    st.subheader("📜 Recent Predictions")

    st.dataframe(
        history.tail(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("📥 Download Prediction History")

    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download CSV Report",
        data=csv,
        file_name="emotion_prediction_history.csv",
        mime="text/csv"
    )

    if st.button(
        "🗑 Clear Prediction History",
        type="secondary"
    ):

        log_file = "logs/emotion_history.csv"

        if os.path.exists(log_file):
            os.remove(log_file)

        st.success("Prediction history cleared successfully.")

        st.rerun()

else:

    st.info("No prediction history available yet. Make your first prediction!")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "Developed using DistilBERT • Hugging Face • Streamlit • Plotly • TensorFlow"
)