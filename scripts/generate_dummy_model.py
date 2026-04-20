import numpy as np
import os

os.makedirs("../models", exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(10, 5)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(3, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")

    X_dummy = np.random.rand(200, 10, 5).astype(np.float32)
    y_dummy = np.random.rand(200, 3).astype(np.float32)
    model.fit(X_dummy, y_dummy, epochs=3, verbose=0)
    model.save("../models/traffic_lstm.h5")
    print("Dummy model saved successfully.")
except Exception as e:
    print(f"TensorFlow not available: {e}")
    print("The system will use rule-based fallback prediction.")
