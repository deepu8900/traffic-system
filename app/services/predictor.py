import numpy as np
import os

MODEL_PATH = "models/traffic_lstm.h5"

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        _model = None
    return _model

def build_input_sequence(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected, history):
    current = [
        min(vehicle_count / 100.0, 1.0),
        density,
        min(speed_kmph / 120.0, 1.0),
        congestion_index,
        1.0 if anomaly_detected else 0.0
    ]
    if len(history) >= 10:
        seq = [[h] + [0, 0, 0, 0] for h in history[-9:]]
        seq.append(current)
    else:
        seq = [current] * 10
    return np.array([seq], dtype=np.float32)

def rule_based_prediction(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected):
    score = 0.0
    score += density * 0.35
    score += congestion_index * 0.35
    score += max(0, (60 - speed_kmph) / 60) * 0.2
    score += (vehicle_count / 80.0) * 0.07
    score += 0.15 if anomaly_detected else 0.0
    score = min(score, 1.0)

    if score < 0.3:
        risk_15 = score * 0.6
        risk_30 = score * 0.75
        risk_45 = score * 0.9
    elif score < 0.6:
        risk_15 = score * 0.85
        risk_30 = score
        risk_45 = min(score * 1.1, 1.0)
    else:
        risk_15 = score
        risk_30 = min(score * 1.05, 1.0)
        risk_45 = min(score * 1.12, 1.0)

    return {
        "jam_probability_15min": round(risk_15, 3),
        "jam_probability_30min": round(risk_30, 3),
        "jam_probability_45min": round(risk_45, 3),
        "overall_risk": round(score, 3),
        "model_used": "rule_based_fallback"
    }

def predict_traffic_jam(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected, history):
    model = load_model()

    if model is None:
        return rule_based_prediction(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected)

    try:
        x = build_input_sequence(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected, history)
        preds = model.predict(x, verbose=0)[0]

        if len(preds) >= 3:
            p15, p30, p45 = float(preds[0]), float(preds[1]), float(preds[2])
        else:
            p15 = p30 = p45 = float(preds[0])

        return {
            "jam_probability_15min": round(p15, 3),
            "jam_probability_30min": round(p30, 3),
            "jam_probability_45min": round(p45, 3),
            "overall_risk": round(max(p15, p30, p45), 3),
            "model_used": "lstm_keras"
        }
    except Exception:
        return rule_based_prediction(vehicle_count, density, speed_kmph, congestion_index, anomaly_detected)
