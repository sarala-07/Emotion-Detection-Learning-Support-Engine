# Emotion-Detection-Learning-Support-Engine
An AI-powered platform that detects student emotions and provides personalized learning support using BiLSTM, BERT, Gemini AI, and Streamlit.
# 🤖 Emotion Detection Learning Support Engine

An AI-powered web application that detects a student's emotional state from text using a fine-tuned DistilBERT model and provides personalized learning guidance through an interactive Streamlit interface.

---

## 🚀 Features

- 🧠 Emotion Detection using DistilBERT
- 📚 Personalized Learning Guidance
- 📊 Interactive Analytics Dashboard
- 📈 Prediction Confidence Visualization
- 📝 Emotion History Logging
- 📥 CSV Report Download
- 🎨 Modern Streamlit User Interface
- ⚡ Fast Real-time Predictions

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| NLP Model | DistilBERT (Hugging Face Transformers) |
| Frontend | Streamlit |
| Data Visualization | Plotly |
| Data Handling | Pandas |
| Model Training | Hugging Face Trainer |

---

## 📂 Project Structure

```
Emotion-Detection-Learning-Support-Engine
│
├── app.py
├── assets/
│   └── banner.jpg
├── style/
│   └── style.css
├── logs/
├── models/
│   └── distilbert_model/
├── utils/
│   ├── advice.py
│   ├── dashboard.py
│   ├── logger.py
│   ├── predict.py
│   └── local_guidance.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/sarala-07/Emotion-Detection-Learning-Support-Engine.git
```

### Navigate into the project

```bash
cd Emotion-Detection-Learning-Support-Engine
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Student enters text.
2. DistilBERT predicts the emotion.
3. Confidence score is calculated.
4. Personalized learning guidance is generated.
5. Prediction is stored in CSV.
6. Dashboard visualizes previous predictions.

---

## 🎯 Supported Emotions

- 😊 Joy
- 😢 Sadness
- 😠 Anger
- 😨 Fear
- ❤️ Love
- 😲 Surprise

---

## 📈 Dashboard

The dashboard provides:

- Emotion Distribution Bar Chart
- Emotion Distribution Pie Chart
- Confidence Trend
- Total Predictions
- Most Common Emotion
- Average Confidence
- CSV Download

---

## 🔮 Future Enhancements

- Gemini AI Integration
- Voice Emotion Detection
- Speech-to-Text Support
- User Login System
- Database Integration
- Multi-language Support
- Dark Mode
- Cloud Deployment

---

## 👨‍💻 Author

**Sarala**

GitHub: https://github.com/sarala-07

---

## ⭐ If you found this project useful, please give it a star!