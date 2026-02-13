import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from src.utils import load_dataset
import os

DATASET_PATH = "dataset/archive (3)"
MODEL_PATH = "models/ser_rnn_lstm.h5"

# Load dataset
X, y = load_dataset(DATASET_PATH)
X = X / np.max(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
emotion_labels = le.classes_

# Load model
model = load_model(MODEL_PATH)

# Predict
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
accuracy = np.mean(y_pred_classes == y_encoded)
print("Test Accuracy:", accuracy)

# Classification report
print(classification_report(y_encoded, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
