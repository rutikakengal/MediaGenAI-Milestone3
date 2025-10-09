import ffmpeg
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# MMS-LID model
MODEL_ID = "facebook/mms-lid-126"

# Load model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

def extract_audio(video_path, output_wav="temp_audio.wav"):
    """Extract audio from video using ffmpeg"""
    (
        ffmpeg
        .input(video_path)
        .output(output_wav, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_wav

def detect_language_from_video(video_path):
    audio_file = extract_audio(video_path)
    speech_array, sampling_rate = sf.read(audio_file)

    # Prepare input
    inputs = extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
    
    # Get model prediction
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    language = model.config.id2label[predicted_id]
    return language

if __name__ == "__main__":
    video_path = "./telugu video.mp4"  # Replace with your video
    language = detect_language_from_video(video_path)
    print(f"[RESULT] Detected Language: {language}")
