import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.utils import load_dataset
from src.product import build_model
import os

# Dataset path
DATASET_PATH = "dataset/archive (3)"

# Load dataset
X, y = load_dataset(DATASET_PATH)
print("Dataset loaded. Shape:", X.shape)

# Normalize
X = X / np.max(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Build model
model = build_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y_categorical.shape[1])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/ser_rnn_lstm.h5")
print("Model saved successfully.")
