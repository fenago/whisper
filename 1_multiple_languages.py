# !pip install openai-whisper

import whisper
import urllib.request

# Download the audio file
AUDIO_URL = "https://github.com/fenago/whisper/raw/refs/heads/main/test_audio_files/dutch_the_netherlands.mp3"
AUDIO_FILE = "dutch_the_netherlands.mp3"

urllib.request.urlretrieve(AUDIO_URL, AUDIO_FILE)

# Load the model
model = whisper.load_model("medium")

def detect_language_and_transcribe(audio_file: str):
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, language_probs = model.detect_language(mel)
    language: str = max(language_probs, key=language_probs.get)  # type: ignore
    print(f"Detected language: {language}")
    options = whisper.DecodingOptions(language=language, task="transcribe")
    result = whisper.decode(model, mel, options)
    print(result)
    return result.text  # type: ignore

# Uncomment to test language detection and transcription
# dutch_test = detect_language_and_transcribe(AUDIO_FILE)

# Uncomment to test basic transcription
# result = model.transcribe(AUDIO_FILE, verbose=True)
# print(result["text"])

# Transcribe and translate Dutch to English
result = model.transcribe(
    AUDIO_FILE,
    verbose=True,
    language="nl",
    task="translate",
)
print(result["text"])
