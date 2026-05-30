# 🎙️ Speech Emotion Recognizer

A deep learning-based web application that detects human emotions from speech audio files. Built using a Dense Neural Network trained on the RAVDESS dataset, with a Flask web interface for real-time predictions.

> 🏫 **Internship Project** — CodeAlpha | Task: Speech Emotion Recognition

---

## 🔍 About the Project

This project analyzes audio files and predicts the emotion of the speaker using extracted MFCC (Mel-Frequency Cepstral Coefficients) features passed through a trained neural network. It supports 8 emotion classes and includes a rule-based fallback system when the model is unavailable.

### Emotions Detected
| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Calm | 😌 |
| Disgust | 🤢 |
| Fearful | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprised | 😲 |

---

## 🗂️ Project Structure

```
SpeechEmotionRecognizer/
│
├── app.py                   # Main Flask application
├── emotion_model.py         # Feature extraction & prediction logic
├── train_model.py           # Model training script
├── main.py                  # CLI-based prediction script
├── audio_test.py            # Audio testing utility
├── requirements.txt         # Python dependencies
│
├── models/                  # Trained model files
│   ├── emotion_model.h5     # Trained Keras model
│   └── label_encoder.pkl    # Label encoder
│
├── templates/
│   └── index.html           # Web UI template
│
├── notebook/
│   └── emotion_recognition.ipynb  # Jupyter notebook
│
├── dataset/                 # ⚠️ Not included (see Dataset section)
└── sample_audio/            # Sample audio for testing
```

---

## ⚙️ Tech Stack

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** Librosa
- **Web Framework:** Flask
- **ML Utilities:** Scikit-learn, Joblib
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

---

## 📦 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

> ⚠️ The dataset is **not included** in this repository due to its large size (~3GB).

**Download Steps:**
1. Download from: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
2. Extract the downloaded zip
3. Place the Actor folders inside your project as:
```
dataset/
└── ravdess/
    ├── Actor_01/
    ├── Actor_02/
    ├── ...
    └── Actor_24/
```
> ⚠️ **Model files not included:** Download pre-trained model files and place them
> in the `models/` folder, or run `python train_model.py` after downloading the dataset.
---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/khilan023/codealpha_SpeechEmotionRecognizer.git
cd codealpha_SpeechEmotionRecognizer
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ Also install ffmpeg on your system:
> - Windows: https://ffmpeg.org/download.html
> - Mac: `brew install ffmpeg`
> - Linux: `sudo apt install ffmpeg`
---

## 🏋️ Train the Model (Optional)

> Skip this step if you already have `models/emotion_model.h5` and `models/label_encoder.pkl`.

After downloading and placing the dataset:
```bash
python train_model.py
```
This will:
- Extract MFCC features from all audio files
- Train a Dense Neural Network for 80 epochs
- Save the model to `models/emotion_model.h5`
- Save the label encoder to `models/label_encoder.pkl`
- Generate confusion matrix and training curves in `models/`

---

## ▶️ Run the Web App

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

Upload a `.wav`, `.mp3`, `.ogg`, `.flac`, or `.m4a` audio file and get instant emotion prediction with confidence scores.

---

## 🖥️ CLI Prediction

To test on a single audio file directly:
```bash
python main.py
```
> Make sure `sample_audio/test.wav` exists or update the file path in `main.py`.

---

## 🧠 How It Works

1. **Audio Input** — User uploads an audio file via the web interface
2. **Feature Extraction** — Librosa extracts 40 MFCC features averaged over time
3. **Prediction** — Features are passed to the trained Dense Neural Network
4. **Fallback** — If model files are missing, a rule-based system using RMS, ZCR, and spectral centroid is used
5. **Output** — Predicted emotion, confidence score, and probability distribution are returned

---

## 📊 Model Architecture

```
Input (40 MFCCs)
    ↓
Dense(256) → BatchNorm → Dropout(0.4)
    ↓
Dense(128) → BatchNorm → Dropout(0.4)
    ↓
Dense(64)  → BatchNorm → Dropout(0.3)
    ↓
Dense(32)  → BatchNorm → Dropout(0.3)
    ↓
Dense(8, softmax) → Predicted Emotion
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 👨‍💻 Author

**Khilan Kaneriya** — Intern at CodeAlpha  
GitHub: [@khilan023](https://github.com/khilan023) · [LinkeDin](https://www.linkedin.com/in/khilan-kaneriya-651883320/)

---

## 📄 License

This project is for educational and internship submission purposes.
