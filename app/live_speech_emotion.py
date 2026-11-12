
import streamlit as st
import io
import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

# ================= PROJECT PATH =================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, BASE_DIR)

from src.utils import extract_features

# ================= SETTINGS =================
SAMPLE_RATE = 22050

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "ser_rnn_lstm.h5"
)

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

# Match the LabelEncoder order used during training
emotion_labels = [
    emotion_map[code] for code in sorted(emotion_map)
]

# ================= LOAD MODEL =================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# ================= STREAMLIT UI =================
st.title("🎤 Live Speech Emotion Recognition")
st.write("Record your voice and predict emotion.")

audio_file = st.audio_input("Record your voice")

if audio_file is not None:

    st.success("Audio recorded successfully!")

    # ================= LOAD AUDIO =================
    audio_signal, sr = librosa.load(
        io.BytesIO(audio_file.getvalue()),
        sr=SAMPLE_RATE
    )

    if len(audio_signal) == 0:
        st.error("No audio detected. Please record again.")
        st.stop()

    # ================= MFCC VISUALIZATION =================
    mfcc = librosa.feature.mfcc(
        y=audio_signal,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mfcc, cmap="coolwarm", ax=ax)
    ax.set_title("MFCC of Recorded Audio")
    st.pyplot(fig)
    plt.close(fig)

    # ================= FEATURE EXTRACTION =================
    features = extract_features(signal=audio_signal)

    max_value = np.max(np.abs(features))

    if max_value == 0:
        st.error("Unable to extract valid audio features.")
        st.stop()

    features = features / max_value
    features = np.expand_dims(features, axis=0)

    # ================= PREDICTION =================
    prediction = model.predict(features, verbose=0)

    predicted_class = np.argmax(prediction, axis=1)[0]

    if len(prediction[0]) != len(emotion_labels):
        st.error("Model output classes do not match emotion labels.")
        st.stop()

    predicted_emotion = emotion_labels[predicted_class]

    st.markdown(
        f"## 🎯 Predicted Emotion: **{predicted_emotion}**"
    )

    # ================= CONFIDENCE CHART =================
    confidences = prediction.flatten()

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        x=emotion_labels,
        y=confidences,
        ax=ax
    )

    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Confidence Scores")

    st.pyplot(fig)
    plt.close(fig)
