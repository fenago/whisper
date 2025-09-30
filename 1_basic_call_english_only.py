# !pip install openai-whisper

import whisper
import urllib.request

# Load the model
MODEL = whisper.load_model("medium.en")

# Download the audio file
AUDIO_URL = "https://github.com/fenago/whisper/raw/refs/heads/main/test_audio_files/terrible_quality.mp3"
AUDIO_FILE = "terrible_quality.mp3"

urllib.request.urlretrieve(AUDIO_URL, AUDIO_FILE)

def get_transcription(audio_file: str):
    result = MODEL.transcribe(audio_file)
    print(result)
    return result

# Run transcription
get_transcription(AUDIO_FILE)
