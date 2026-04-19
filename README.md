# Speech Emotion Recognition

## What This Project Does
This project detects human emotions from speech audio using deep learning.

It provides:
- A training pipeline to build an SER model from the RAVDESS-style dataset in `dataset/archive (3)`.
- A testing script to evaluate model accuracy and generate a confusion matrix.
- A live Streamlit app that records microphone audio and predicts emotion in real time.

The model uses MFCC-based audio features and a hybrid SimpleRNN + LSTM architecture for classification.

## My Contribution
I designed and implemented the full Speech Emotion Recognition workflow, including:
- Audio preprocessing and MFCC feature extraction (`src/utils.py`).
- Dataset loading and emotion label preparation from `.wav` filenames.
- Deep learning model architecture using `SimpleRNN` + `LSTM` (`src/product.py`).
- Model training and saving pipeline (`train.py`).
- Model evaluation with classification report and confusion matrix visualization (`test.py`).
- Real-time inference app with microphone recording and confidence score chart (`app/live_speech_emotion.py`).

## Tech Stack Used
- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- librosa
- matplotlib
- seaborn
- Streamlit
- sounddevice
- SciPy

## Project Structure
```text
Speech_Emotion_Recognition/
|-- app/
|   `-- live_speech_emotion.py
|-- src/
|   |-- utils.py
|   `-- product.py
|-- models/
|   `-- ser_rnn_lstm.h5
|-- dataset/
|   `-- archive (3)/
|-- train.py
|-- test.py
`-- requirements.txt
```

## How To Run
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train model
```bash
python train.py
```

### 3) Evaluate model
```bash
python test.py
```

### 4) Run live app
```bash
streamlit run app/live_speech_emotion.py
```

## Output
- Trained model saved at `models/ser_rnn_lstm.h5`
- Console metrics: accuracy + classification report
- Confusion matrix visualization
- Live emotion prediction and confidence scores in Streamlit


