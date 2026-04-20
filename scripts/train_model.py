import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

df = pd.read_csv("/kaggle/input/metro-interstate-traffic-volume/Metro_Interstate_Traffic_Volume.csv")
print("Dataset shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)
df = df.drop_duplicates(subset="date_time")

df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_peak_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
df["is_peak_evening"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)

weather_map = {w: i for i, w in enumerate(df["weather_main"].unique())}
df["weather_code"] = df["weather_main"].map(weather_map)

feature_cols = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_code",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_morning",
    "is_peak_evening"
]

df = df[feature_cols].dropna()
print("Clean dataset shape:", df.shape)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[feature_cols])

traffic_idx = feature_cols.index("traffic_volume")

TIMESTEPS = 24
FEATURES = len(feature_cols)

def make_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps - 3):
        seq = data[i:i+timesteps]
        p15 = data[i+timesteps, traffic_idx]
        p30 = data[i+timesteps+1, traffic_idx]
        p45 = data[i+timesteps+2, traffic_idx] if i+timesteps+2 < len(data) else data[i+timesteps+1, traffic_idx]
        X.append(seq)
        y.append([p15, p30, p45])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

print("Building sequences...")
X, y = make_sequences(scaled, TIMESTEPS)
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def build_model(timesteps, features):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(timesteps, features)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(96, return_sequences=True),
        BatchNormalization(),
        Dropout(0.25),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(48, activation="relu"),
        Dense(24, activation="relu"),
        Dense(3, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )
    return model

model = build_model(TIMESTEPS, FEATURES)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint("/kaggle/working/traffic_lstm.h5", monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=5, min_lr=1e-6, verbose=1)
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.title("MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=120)
plt.show()

print("\nEvaluating on test set...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (Huber): {test_loss:.5f}")
print(f"Test MAE:          {test_mae:.5f}")

preds = model.predict(X_test[:10], verbose=0)
print("\nSample predictions (15min | 30min | 45min jam probability):")
for i, p in enumerate(preds):
    actual = y_test[i]
    print(f"  Sample {i+1}: pred=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  actual=[{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}]")

meta = {
    "timesteps": TIMESTEPS,
    "features": FEATURES,
    "feature_cols": feature_cols,
    "traffic_col_index": traffic_idx,
    "weather_map": weather_map,
    "test_loss": float(test_loss),
    "test_mae": float(test_mae),
    "epochs_trained": len(history.history["loss"]),
    "dataset": "Metro Interstate Traffic Volume"
}

with open("/kaggle/working/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

scaler_params = {
    "scale_": scaler.scale_.tolist(),
    "min_": scaler.min_.tolist(),
    "data_min_": scaler.data_min_.tolist(),
    "data_max_": scaler.data_max_.tolist(),
    "data_range_": scaler.data_range_.tolist()
}
with open("/kaggle/working/scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

print("\n--- Training Complete ---")
print("Files saved to /kaggle/working/:")
print("  traffic_lstm.h5       <- your trained model, download this")
print("  model_meta.json       <- model metadata")
print("  scaler_params.json    <- scaler values for inference")
print("  training_curves.png   <- loss/mae plots")
print(f"\nFinal best val_loss: {min(history.history['val_loss']):.5f}")
