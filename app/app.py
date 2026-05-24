from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import joblib

from tensorflow.keras.models import load_model


app = Flask(__name__)


# Load model
model = load_model("../models/emotion_model.h5")


# Load encoder
encoder = joblib.load("../models/label_encoder.pkl")


# Upload folder
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Feature extraction
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


@app.route("/", methods=["GET", "POST"])
def index():

    emotion = None

    if request.method == "POST":

        file = request.files["audio"]

        if file:

            file_path = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(file_path)

            feature = extract_features(file_path)

            feature = np.expand_dims(feature, axis=0)

            prediction = model.predict(feature)

            predicted_label = np.argmax(prediction)

            emotion = encoder.inverse_transform(
                [predicted_label]
            )[0]

    return render_template(
        "index.html",
        emotion=emotion
    )


if __name__ == "__main__":

    app.run(debug=True)