import torch
import sounddevice as sd
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ---------------------------
# Configuration
# ---------------------------
MODEL_ID = "facebook/mms-lid-126"  # Pretrained multilingual language ID model
DURATION =5   # seconds to record 
SR = 16000     # Sampling rate

# Whitelist your 12 Indian languages + English
WHITELIST_LANGUAGES = {
    "eng": "English",
    "hin": "Hindi",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "ben": "Bengali",
    "mar": "Marathi",
    "guj": "Gujarati",
    "pan": "Punjabi",
    "ori": "Odia",
    "urd": "Urdu"
}

# ---------------------------
# Load model and feature extractor
# ---------------------------
print("[INFO] Loading MMS-LID model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()
print("[INFO] Model loaded!")

# ---------------------------
# Record audio from microphone
# ---------------------------
def record_audio(duration=DURATION, sr=SR):
    print(f"[INFO] Recording {duration}s...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio

# ---------------------------
# Predict language
# ---------------------------
def predict_language(audio_array, sr=SR):
    inputs = feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]
    language = WHITELIST_LANGUAGES.get(predicted_label, "Unknown")
    return language

# ---------------------------
# Main loop for real-time detection
# ---------------------------
if __name__ == "__main__":
    try:
        while True:
            audio_data = record_audio()
            language = predict_language(audio_data)
            print("----------------------------------------")
            print(f"[RESULT] Detected Language: {language}")
            print("----------------------------------------")
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")


