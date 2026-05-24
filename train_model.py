"""
train_model.py
==============
Train an LSTM model for Speech Emotion Recognition on the RAVDESS dataset.

Steps:
  1. pip install -r requirements.txt
  2. Download RAVDESS: https://zenodo.org/record/1188976
     Extract into:  data/RAVDESS/Actor_*/
  3. Run:  python train_model.py
  4. Models saved to: models/emotion_model.h5  &  models/label_encoder.pkl
"""

import os
import glob
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = "dataset/ravdess"   # folder with Actor_01/ ... Actor_24/
MODEL_DIR   = "models"
FEATURE_DIM = 40               # 40 MFCCs — must match emotion_model.py
EPOCHS      = 80
BATCH_SIZE  = 32
RANDOM_SEED = 42

# RAVDESS emotion code → label  (must match existing label_encoder.pkl naming)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Feature extraction (must match emotion_model.py) ─────────────────────────

def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, duration=3.0, offset=0.5)

    # 40 MFCCs averaged over time → shape (40,)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


# ─── Load dataset ─────────────────────────────────────────────────────────────

def load_ravdess(data_dir: str):
    X, y = [], []
    audio_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    print(f"Found {len(audio_files)} audio files in {data_dir}")

    for fp in audio_files:
        try:
            # RAVDESS filename format: 03-01-NN-...wav  (3rd segment = emotion code)
            parts = os.path.basename(fp).split("-")
            emotion_code = parts[2]
            if emotion_code not in EMOTION_MAP:
                continue
            label = EMOTION_MAP[emotion_code]

            features = extract_features(fp)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Loaded {len(X)} samples across {len(set(y))} emotion classes.")
    return np.array(X), np.array(y)


# ─── Build Dense model ────────────────────────────────────────────────────────

def build_model(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    model = Sequential([
        Dense(256, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Main training loop ───────────────────────────────────────────────────────

def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    X, y_raw = load_ravdess(DATA_DIR)
    if len(X) == 0:
        print("\n❌ No data found. Check DATA_DIR path.")
        return

    # 2. Encode labels
    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y_raw)
    y_cat   = to_categorical(y_enc)
    n_classes = len(encoder.classes_)
    print(f"Classes: {list(encoder.classes_)}")

    # 3. Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # 4. Build model
    model = build_model(input_shape=(FEATURE_DIM,), n_classes=n_classes)
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "emotion_model.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
    ]

    # 6. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # 7. Save label encoder
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"\n✅ Model saved to {MODEL_DIR}/emotion_model.h5")
    print(f"✅ Encoder saved to {MODEL_DIR}/label_encoder.pkl")

    # 8. Evaluate
    y_pred_raw = model.predict(X_val, verbose=0)
    y_pred     = np.argmax(y_pred_raw, axis=1)
    y_true     = np.argmax(y_val, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    # 9. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_,
                yticklabels=encoder.classes_, cmap="Purples")
    plt.title("Confusion Matrix — Speech Emotion Recognition")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print(f"✅ Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")

    # 10. Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["accuracy"],     label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print(f"✅ Training curves saved to {MODEL_DIR}/training_curves.png")


if __name__ == "__main__":
    main()