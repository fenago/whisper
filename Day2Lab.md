# Whisper Language Detection & Transcription Lab

## Learning Objectives
By the end of this exercise, you will be able to:
- Install and set up OpenAI Whisper for audio processing
- Detect languages in audio files with confidence scoring
- Transcribe audio in the detected language
- Translate audio to English
- Implement error handling for low-confidence predictions

---

## Prerequisites
- Python 3.8 or higher installed
- Basic understanding of Python functions
- Familiarity with try-except error handling

---

## Part 1: Setup and Installation (10 minutes)

### Step 1: Install Whisper
Open your terminal or notebook and run:
```python
!pip install openai-whisper
```

### Step 2: Import Required Libraries
```python
import whisper
import urllib.request
```

**💡 Checkpoint Question:** What does the `urllib.request` module do?

---

## Part 2: Download Test Audio (5 minutes)

### Step 3: Download the Audio File
Create variables for the audio URL and filename:
```python
AUDIO_URL = "https://github.com/fenago/whisper/raw/refs/heads/main/test_audio_files/dutch_the_netherlands.mp3"
AUDIO_FILE = "dutch_the_netherlands.mp3"
```

### Step 4: Retrieve the File
```python
urllib.request.urlretrieve(AUDIO_URL, AUDIO_FILE)
```

**✅ Task:** Verify the file downloaded by checking your current directory.

---

## Part 3: Load the Whisper Model (5 minutes)

### Step 5: Load the Model
```python
model = whisper.load_model("medium")
```

**💡 Discussion Point:** Whisper offers different model sizes: `tiny`, `base`, `small`, `medium`, `large`. What's the tradeoff between model sizes?

---

## Part 4: Build the Language Detection Function (20 minutes)

### Step 6: Create the Function Signature
Start by defining your function with type hints:
```python
def detect_language_and_transcribe(audio_file: str, confidence_threshold: float = 0.5):
    """
    Detect language and transcribe audio with confidence checking.
    
    Args:
        audio_file: Path to the audio file
        confidence_threshold: Minimum confidence required (default: 0.5)
    
    Returns:
        Tuple of (transcribed_text, detected_language, confidence)
    """
```

**✅ Task:** What does the default value `0.5` mean for `confidence_threshold`?

### Step 7: Load and Prepare Audio
Inside the function, add a try-except block and load the audio:
```python
    try:
        # Load and prepare audio
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
```

**💡 Checkpoint Question:** What is a mel spectrogram and why do we use it?

### Step 8: Detect the Language
Add language detection logic:
```python
        # Detect language
        _, language_probs = model.detect_language(mel)
        detected_language: str = max(language_probs, key=language_probs.get)
        confidence: float = language_probs[detected_language]
```

**✅ Task:** What does `language_probs` contain? What type of data structure is it?

### Step 9: Display Detection Results
Add clear output formatting:
```python
        # Clear printout of language detection
        print("=" * 50)
        print(f"🎯 LANGUAGE DETECTION RESULT")
        print("=" * 50)
        print(f"Detected Language: {detected_language.upper()}")
        print(f"Confidence Score: {confidence:.2%}")
        print("=" * 50)
```

**💡 Discussion:** What does `{confidence:.2%}` formatting do?

### Step 10: Implement Confidence Checking
Add logic to check if confidence meets the threshold:
```python
        # Check confidence threshold
        if confidence < confidence_threshold:
            error_msg = (
                f"⚠️  Low confidence warning: Language detection confidence "
                f"({confidence:.2%}) is below threshold ({confidence_threshold:.2%}). "
                f"Detected language '{detected_language}' may be incorrect."
            )
            print(error_msg)
            raise ValueError(error_msg)
```

**✅ Task:** Why do we `raise ValueError` here instead of just printing a warning?

### Step 11: Transcribe the Audio
Add the transcription logic:
```python
        # Transcribe with detected language
        print(f"\n📝 Transcribing in {detected_language}...\n")
        options = whisper.DecodingOptions(language=detected_language, task="transcribe")
        result = whisper.decode(model, mel, options)
        
        return result.text, detected_language, confidence
```

### Step 12: Add Error Handling
Complete the function with exception handling:
```python
    except ValueError as e:
        # Re-raise confidence errors
        raise e
    except Exception as e:
        error_msg = f"❌ Error during language detection or transcription: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
```

**💡 Discussion:** Why do we handle `ValueError` and `Exception` separately?

---

## Part 5: Test Your Function (15 minutes)

### Step 13: Example 1 - Basic Language Detection
Test your function with error handling:
```python
print("\n" + "="*50)
print("EXAMPLE 1: Language Detection with Confidence Check")
print("="*50 + "\n")

try:
    text, language, confidence = detect_language_and_transcribe(
        AUDIO_FILE, 
        confidence_threshold=0.5
    )
    print(f"✅ Transcription successful!")
    print(f"Text: {text}")
except ValueError as e:
    print(f"⚠️  Continuing despite low confidence: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

**✅ Task:** Run this code. What language was detected? What was the confidence score?

### Step 14: Example 2 - Translation to English
Test the translation capability:
```python
print("\n" + "="*50)
print("EXAMPLE 2: Transcription with Translation to English")
print("="*50 + "\n")

try:
    result = model.transcribe(
        AUDIO_FILE,
        verbose=True,
        language="nl",  # Explicitly set Dutch
        task="translate",  # Translate to English
    )
    print("\n" + "="*50)
    print("📄 TRANSLATION RESULT")
    print("="*50)
    print(f"Original Language: Dutch (nl)")
    print(f"Translated Text: {result['text']}")
    print("="*50)
except Exception as e:
    print(f"❌ Translation error: {e}")
```

**💡 Discussion:** What's the difference between `task="transcribe"` and `task="translate"`?

### Step 15: Example 3 - Show Top Predictions
Display the top 3 language predictions:
```python
print("\n" + "="*50)
print("EXAMPLE 3: Top Language Predictions")
print("="*50 + "\n")

try:
    audio = whisper.load_audio(AUDIO_FILE)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, language_probs = model.detect_language(mel)
    
    # Sort and display top 3
    sorted_languages = sorted(language_probs.items(), key=lambda x: x[1], reverse=True)
    print("Top 3 Language Predictions:")
    for i, (lang, prob) in enumerate(sorted_languages[:3], 1):
        print(f"  {i}. {lang.upper()}: {prob:.2%}")
    print("="*50)
except Exception as e:
    print(f"❌ Error showing predictions: {e}")
```

**✅ Task:** What are the top 3 predicted languages and their probabilities?

---

## Part 6: Experimentation (20 minutes)

### Challenge 1: Test Different Audio Files
Download and test another audio file from a different language.

**Hint:** You can find other test files at:
- `german_germany.mp3`
- `spanish_spain.mp3`
- `french_france.mp3`

### Challenge 2: Adjust Confidence Threshold
Experiment with different confidence thresholds (0.3, 0.7, 0.9). What happens?

### Challenge 3: Add Timestamp Support
Modify the code to show timestamps for each segment. Research the `word_timestamps` parameter.

### Challenge 4: Create a Multi-File Processor
Write a function that processes multiple audio files in a batch and generates a summary report.

---

## Reflection Questions

1. **When would low confidence detection be a problem in real-world applications?**

2. **Why might Whisper confuse similar languages (e.g., Dutch and Afrikaans)?**

3. **What are the ethical considerations when using automatic transcription for:
   - Medical records
   - Legal proceedings
   - Educational assessments**

4. **How could you improve the accuracy of language detection?**

---

## Submission Checklist

- [ ] All code runs without errors
- [ ] Function correctly detects language with confidence score
- [ ] Successfully transcribes audio in detected language
- [ ] Successfully translates to English
- [ ] Displays top 3 language predictions
- [ ] Answered all checkpoint questions
- [ ] Completed at least one challenge

---

## Additional Resources

- [Whisper GitHub Repository](https://github.com/openai/whisper)
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md)
- [Understanding Mel Spectrograms](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

---

## Common Troubleshooting

**Error: "No module named 'whisper'"**
- Solution: Make sure you ran `!pip install openai-whisper`

**Error: "CUDA out of memory"**
- Solution: Try using a smaller model (`tiny` or `base`)

**Low confidence scores**
- Check audio quality (background noise, clarity)
- Try a larger model size
- Ensure the audio file isn't corrupted
