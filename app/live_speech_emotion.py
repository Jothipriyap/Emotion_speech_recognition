import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import extract_features

# ================= SETTINGS =================
SAMPLE_RATE = 22050
DURATION = 4
MODEL_PATH = "../models/ser_rnn_lstm.h5"
DATASET_PATH = "../dataset/archive (3)"

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)

# ================= EMOTION MAPPING =================
emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

# ================= LABEL ENCODER =================
labels = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion_name = emotion_map.get(emotion_code)
            if emotion_name:
                labels.append(emotion_name)

le = LabelEncoder()
le.fit(labels)
emotion_labels = le.classes_

# ================= STREAMLIT UI =================
st.title("🎤 Live Speech Emotion Recognition")
st.write("Record your voice and predict emotion.")

if st.button("Record Audio"):
    st.write("Recording...")
    
    recording = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)
    sd.wait()

    st.success("Recording Complete!")

    audio_signal = recording.flatten()

    # ================= MFCC VISUALIZATION =================
    mfcc = librosa.feature.mfcc(y=audio_signal,
                                sr=SAMPLE_RATE,
                                n_mfcc=40)

    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc, cmap='coolwarm')
    plt.title("MFCC of Recorded Audio")
    st.pyplot(plt)

    # ================= FEATURE EXTRACTION =================
    features = extract_features(signal=audio_signal)
    features = features / np.max(features)
    features = np.expand_dims(features, axis=0)

    # ================= PREDICTION =================
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    predicted_emotion = le.inverse_transform([predicted_class])

    # Display Emotion Clearly
    st.markdown(f"## 🎯 Predicted Emotion: **{predicted_emotion}**")
    

    # ================= CONFIDENCE CHART =================
    confidences = prediction.flatten()

    fig, ax = plt.subplots()
    sns.barplot(x=emotion_labels, y=confidences, ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Probability")
    plt.title("Emotion Confidence Scores")
    st.pyplot(fig)
