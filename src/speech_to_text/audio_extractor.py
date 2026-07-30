from pathlib import Path
import subprocess

from .config import AUDIOS_FOLDER


def has_audio_stream(video_path: Path) -> bool:
    """
    Check whether the video contains an audio stream.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() == "audio"


def extract_audio(
    video_path: Path,
    video_id: str,
) -> Path:
    """
    Extract audio from a video and save it as MP3.
    """
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    audio_path = (
        AUDIOS_FOLDER
        / f"{video_id}.mp3"
    )

    if audio_path.exists():
        return audio_path

    if not has_audio_stream(video_path):
        raise ValueError(
            f"{video_id} does not contain an audio stream."
        )

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        "128k",
        "-y",
        str(audio_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file was not created: {audio_path}"
        )

    return audio_path