from pathlib import Path

from src.speech_to_text.transcriber import transcribe_media


audio_path = Path(
    "data/audios/test_video.mp3"
)

result = transcribe_media(audio_path)

print("\nTranscription completed successfully!")
print(f"Transcript: {result['transcript']}")
print(f"Detected language: {result['detected_language']}")
print(f"Language probability: {result['language_probability']}")
print(f"Audio duration: {result['audio_duration_seconds']} seconds")