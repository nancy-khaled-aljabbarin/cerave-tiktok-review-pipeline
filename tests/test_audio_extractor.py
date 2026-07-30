from pathlib import Path

from src.speech_to_text.audio_extractor import extract_audio


video_path = Path(
    "data/videos/test_video.mp4"
)

audio_path = extract_audio(
    video_path=video_path,
    video_id="test_video",
)

print("Audio extracted successfully!")
print(f"Saved at: {audio_path}")