import os
import glob
import csv
import datetime
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment

# ---------- Config ----------
RAW_ENGLISH = "raw_data/english"
RAW_HINDI = "raw_data/hindhi"
DATASET_DIR = "dataset"
CHUNK_MS = 5000  # 5 seconds

recognizer = sr.Recognizer()

def ensure_dirs(lang_name):
    """Make sure dataset folders exist"""
    base = os.path.join(DATASET_DIR, lang_name)
    os.makedirs(os.path.join(base, "audio_chunks"), exist_ok=True)
    os.makedirs(os.path.join(base, "translated_audio"), exist_ok=True)
    return base

def split_text(text, max_length=4500):
    """Split text into chunks for translator limits"""
    chunks = []
    while len(text) > max_length:
        split_point = text.rfind(" ", 0, max_length)
        if split_point == -1:
            split_point = max_length
        chunks.append(text[:split_point])
        text = text[split_point:]
    chunks.append(text)
    return chunks

def process_audio_file(file_path, src_lang, tgt_lang, lang_name):
    base = ensure_dirs(lang_name)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # CSV file to log transcriptions
    csv_file = os.path.join(base, "transcriptions.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_file", "original_text", "translated_text"])

    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"⚠️ Could not load {file_path}: {e}")
        return

    chunks = len(audio) // CHUNK_MS + (1 if len(audio) % CHUNK_MS else 0)

    for i in range(chunks):
        start = i * CHUNK_MS
        end = min((i + 1) * CHUNK_MS, len(audio))
        chunk = audio[start:end]

        chunk_filename = f"{filename}_{ts}_chunk{i+1}.wav"
        chunk_path = os.path.join(base, "audio_chunks", chunk_filename)
        chunk.export(chunk_path, format="wav")

        original_text = ""
        translated_text = ""

        try:
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                if src_lang == "en":
                    original_text = recognizer.recognize_google(audio_data, language="en-IN")
                else:
                    original_text = recognizer.recognize_google(audio_data, language="hi-IN")
        except sr.UnknownValueError:
            original_text = ""
        except Exception as e:
            print(f"⚠️ STT failed on {chunk_filename}: {e}")
            continue

        if original_text.strip():
            try:
                for text_part in split_text(original_text.strip()):
                    translated_text += " " + GoogleTranslator(source=src_lang, target=tgt_lang).translate(text_part)
            except Exception as e:
                print(f"⚠️ Translation failed on {chunk_filename}: {e}")
                continue

            # Save translated audio
            try:
                tts_out = os.path.join(base, "translated_audio", chunk_filename.replace(".wav", ".mp3"))
                gTTS(text=translated_text.strip(), lang=tgt_lang).save(tts_out)
            except Exception as e:
                print(f"⚠️ TTS failed on {chunk_filename}: {e}")

        # Append to CSV
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([chunk_filename, original_text.strip(), translated_text.strip()])

        print(f"✅ Processed {chunk_filename}")

def build_dataset():
    # English → Hindi
    for f in glob.glob(os.path.join(RAW_ENGLISH, "*.mp3")) + glob.glob(os.path.join(RAW_ENGLISH, "*.wav")):
        process_audio_file(f, src_lang="en", tgt_lang="hi", lang_name="english")

    # Hindi → English
    for f in glob.glob(os.path.join(RAW_HINDI, "*.mp3")) + glob.glob(os.path.join(RAW_HINDI, "*.wav")):
        process_audio_file(f, src_lang="hi", tgt_lang="en", lang_name="hindhi")

if __name__ == "__main__":
    build_dataset()
