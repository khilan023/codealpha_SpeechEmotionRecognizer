import numpy as np
import librosa
import os

# ─── Model paths ──────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "emotion_model.h5")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")

_model   = None
_encoder = None


def _load_model():
    global _model, _encoder
    if _model is not None:
        return True
    try:
        from tensorflow.keras.models import load_model
        import joblib
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            _model   = load_model(MODEL_PATH)
            _encoder = joblib.load(ENCODER_PATH)
            print(f"[emotion_model] Model loaded OK  Classes: {list(_encoder.classes_)}")
            print(f"[emotion_model] Model input shape: {_model.input_shape}")
            return True
        else:
            print(f"[emotion_model] Model files not found at {MODEL_PATH} / {ENCODER_PATH}. Using rule-based fallback.")
    except Exception as e:
        print(f"[emotion_model] Could not load model ({e}). Using rule-based fallback.")
    return False


# ─── Eagerly pre-load model at import time ────────────────────────────────────
print("[emotion_model] Pre-loading TensorFlow model (this may take a moment)...")
_model_ready = _load_model()
print(f"[emotion_model] Pre-load complete. Model ready: {_model_ready}")


# ─── Feature extraction ───────────────────────────────────────────────────────
# FIX: Do NOT use offset when loading — short audio files become empty arrays
# when offset >= audio duration. Instead load the full file, then trim/pad to
# a fixed length so the model always gets a consistent input shape.

def extract_features(file_path: str, target_duration: float = 3.0) -> np.ndarray:
    # Load the full audio file without offset so we don't accidentally skip it
    y, sr = librosa.load(file_path, duration=None, offset=0.0)

    # Guard: if the file is empty, raise early with a clear error
    if len(y) == 0:
        raise ValueError("Audio file is empty (no samples loaded). Please upload a valid audio file.")

    # If the clip is shorter than target_duration, pad with zeros (silence)
    target_samples = int(target_duration * sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    else:
        y = y[:target_samples]

    # 40 MFCCs averaged over time → shape (40,)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # (40,)

    # Guard: NaN values will silently break the model
    if np.any(np.isnan(mfcc_mean)) or np.any(np.isinf(mfcc_mean)):
        raise ValueError("Feature extraction produced invalid values (NaN/Inf). The audio may be silent or corrupt.")

    return mfcc_mean


# ─── Emotion labels ───────────────────────────────────────────────────────────
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

EMOTION_DISPLAY = {
    "angry":     {"emoji": "😠", "desc": "High arousal negative emotion detected."},
    "calm":      {"emoji": "😌", "desc": "The speaker sounds calm and composed."},
    "disgust":   {"emoji": "🤢", "desc": "Negative reaction detected in the voice."},
    "fearful":   {"emoji": "😨", "desc": "Signs of anxiety or fear in the voice."},
    "happy":     {"emoji": "😊", "desc": "The speaker sounds joyful and positive."},
    "neutral":   {"emoji": "😐", "desc": "Calm, even tone with no strong emotion."},
    "sad":       {"emoji": "😢", "desc": "Detected sadness or low-energy mood."},
    "surprised": {"emoji": "😲", "desc": "Sudden or unexpected emotional spike."},
}


# ─── Rule-based fallback ──────────────────────────────────────────────────────

def _rule_based_predict(file_path: str) -> dict:
    try:
        # FIX: same safe loading — no offset, pad short clips
        y, sr = librosa.load(file_path, duration=None, offset=0.0)

        if len(y) == 0:
            raise ValueError("Empty audio")

        target_samples = int(3.0 * sr)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode="constant")
        else:
            y = y[:target_samples]

        rms    = float(np.mean(librosa.feature.rms(y=y)))
        zcr    = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        if rms > 0.08 and zcr > 0.08:
            emotion, conf = "angry", 72
        elif rms > 0.06:
            emotion, conf = "happy", 68
        elif rms < 0.02:
            emotion, conf = "sad", 65
        elif zcr > 0.10:
            emotion, conf = "fearful", 60
        elif spectral_centroid > 3000 and rms > 0.05:
            emotion, conf = "surprised", 63
        else:
            emotion, conf = "neutral", 70

        base_pct = round((100 - conf) / (len(EMOTIONS) - 1), 1)
        probs = {e: base_pct for e in EMOTIONS}
        probs[emotion] = float(conf)

        return {"emotion": emotion, "confidence": conf, "probabilities": probs}

    except Exception as e:
        print(f"[emotion_model] Rule-based error: {e}")
        return {
            "emotion": "neutral",
            "confidence": 55,
            "probabilities": {e: round(100 / len(EMOTIONS), 1) for e in EMOTIONS}
        }


# ─── Main prediction entry point ─────────────────────────────────────────────

def predict_emotion(audio_path: str) -> dict:
    if _load_model():
        try:
            features = extract_features(audio_path)
            features = np.expand_dims(features, axis=0)

            raw_pred = _model.predict(features, verbose=0)[0]

            predicted_idx   = int(np.argmax(raw_pred))
            predicted_label = _encoder.inverse_transform([predicted_idx])[0].lower()
            confidence      = int(round(float(raw_pred[predicted_idx]) * 100))

            probs = {
                label.lower(): round(float(raw_pred[i]) * 100, 1)
                for i, label in enumerate(_encoder.classes_)
            }

            return {
                "emotion":       predicted_label,
                "confidence":    confidence,
                "probabilities": probs,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[emotion_model] Inference error: {e}. Falling back to rule-based.")

    return _rule_based_predict(audio_path)