import librosa
import numpy as np
import os

MAX_PAD_LEN = 200

def extract_features(file_path=None, signal=None):
    """
    Extract MFCC from either a file or a raw signal
    """
    if file_path:
        signal, sr = librosa.load(file_path, sr=22050)
    else:
        sr = 22050

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc.T  # shape: (time_steps, features)

def load_dataset(dataset_path):
    features = []
    labels = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                emotion = file.split("-")[2]
                features.append(extract_features(file_path=path))
                labels.append(emotion)

    return np.array(features), np.array(labels)
