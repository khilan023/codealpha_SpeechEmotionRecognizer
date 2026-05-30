from flask import Flask, render_template, request, jsonify
import os
import traceback
from emotion_model import predict_emotion

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check file present
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]

    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use WAV, MP3, OGG, FLAC or M4A"}), 400

    filepath = None
    wav_path = None

    try:
        # 2. Save original file
        safe_name = os.path.basename(file.filename)
        filepath  = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)

        # 3. Convert to WAV if needed (pydub handles all formats)
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext == "wav":
            wav_path = filepath          # already WAV, no conversion needed
        else:
            from pydub import AudioSegment
            audio    = AudioSegment.from_file(filepath)
            wav_path = filepath + ".wav"
            audio.export(wav_path, format="wav")

        # 4. Run emotion prediction
        print(f"[app] Predicting emotion for: {wav_path}")
        result = predict_emotion(wav_path)
        print(f"[app] Prediction result: {result}")

        # 5. Ensure required keys exist (safety net)
        if "emotion" not in result:
            result["emotion"] = "neutral"
        if "confidence" not in result:
            result["confidence"] = 50
        if "probabilities" not in result:
            result["probabilities"] = {}

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()                          # logs full error to terminal
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # 6. Clean up temp files
        if wav_path and wav_path != filepath and os.path.exists(wav_path):
            try: os.remove(wav_path)
            except: pass
        if filepath and os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass


if __name__ == "__main__":
    app.run(debug=False)
