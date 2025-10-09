## 🗣️ Indian Language Detection & Transcription

This project detects spoken languages from microphone input or video files and extracts the text in the corresponding language. It supports 12 Indian languages using a combination of MMS-LID (for language detection) and OpenAI Whisper (for transcription).

## 🔹 Supported Languages
| Language  | Code |
| --------- | ---- |
| English   | eng  |
| Hindi     | hin  |
| Tamil     | tam  |
| Telugu    | tel  |
| Kannada   | kan  |
| Malayalam | mal  |
| Bengali   | ben  |
| Marathi   | mar  |
| Gujarati  | guj  |
| Punjabi   | pan  |
| Odia      | ori  |
| Urdu      | urd  |

## ⚙️ Features

• Real-time language detection from microphone input.

• Audio extraction from video files and detection of the language.

• Accurate text transcription in the detected language using Whisper.

• Easy to extend for other Indic languages.

• Supports GPU acceleration for faster transcription (if available).

## 🛠️ Technologies Used

• Python 3

• PyTorch

• Transformers (HuggingFace) – MMS-LID model for language detection

• Whisper (OpenAI) – Audio-to-text transcription

• SoundDevice – Recording audio from microphone

• ffmpeg – Audio extraction from video files

• SoundFile / SciPy – Audio file handling

## 🚀 Getting Started
1. Clone the repository
   ```
   git clone https://github.com/rutikakengal/MediaGenAI Milestone 3.git
   cd indian-language-detector
   ```
2. Create & activate virtual environment
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / Mac
   source .venv/bin/activate
   ```
3. Install dependencies
    ```
    pip install -r requirements.txt
    ```
4. Run Microphone-based Detection
    ```
   python language_detector.py
    ```
##
### Speak into the microphone for ~10 seconds. The script will:
  • Detect the language.
  
  • Transcribe the spoken text in the detected language.
##
5. Run Video-based Detection
   ```
   python video_detector.py
   ```
##
### Replace example_video.mp4 with your video. The script will:

• Extract audio from video.

• Detect the language.

Transcribe the audio to text.
##

### ⚡ Notes

• For better transcription, a GPU is recommended when using Whisper.

• Supported recording duration is 10 seconds, but you can modify it in the scripts.

• Only 12 Indian languages are whitelisted for accurate detection.
##
### 📌 References

• MMS-LID model: HuggingFace MMS-LID

• Whisper: OpenAI Whisper

• ffmpeg: FFmpeg Python
##
### ⭐ Future Improvements

• Add real-time video streaming language detection.

• Extend more Indian languages.

• Implement GUI interface for easy usage.

• Faster transcription using GPU acceleration.
##
### 🌟 Loved this project?

• If you enjoyed this project, show some love and give it a ⭐ on GitHub!

• It helps me keep improving and adding new features.

##
