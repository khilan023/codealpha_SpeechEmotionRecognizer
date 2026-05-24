import librosa
import numpy as np
import joblib

from tensorflow.keras.models import load_model


model = load_model("models/emotion_model.h5")


encoder = joblib.load("models/label_encoder.pkl")


def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


file_path = "sample_audio/test.wav"


feature = extract_features(file_path)


feature = np.expand_dims(feature, axis=0)


prediction = model.predict(feature)


predicted_label = np.argmax(prediction)


emotion = encoder.inverse_transform([predicted_label])


print("Predicted Emotion:", emotion[0])