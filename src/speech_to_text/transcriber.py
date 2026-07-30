from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from .config import (
    MODEL_NAME,
    DEVICE,
    COMPUTE_TYPE,
)


_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    """
    Load the Whisper model once and reuse it
    for all audio files.
    """
    global _model

    if _model is None:
        print(f"Loading Whisper model: {MODEL_NAME}")

        _model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

    return _model


def transcribe_media(media_path: Path) -> dict:
    """
    Convert spoken audio into text.
    """
    model = get_model()

    segments, information = model.transcribe(
        str(media_path),
        beam_size=5,
        vad_filter=False,
    )

    transcript_parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            transcript_parts.append(text)

    transcript = " ".join(transcript_parts)

    return {
        "transcript": transcript,
        "detected_language": information.language,
        "language_probability": round(
            information.language_probability,
            4,
        ),
        "audio_duration_seconds": round(
            information.duration,
            2,
        ),
    }