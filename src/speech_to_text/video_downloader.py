from pathlib import Path
import subprocess

from yt_dlp import YoutubeDL

from .config import VIDEOS_FOLDER


def convert_to_h264(
    source_path: Path,
    final_path: Path,
) -> None:
    """
    Convert a downloaded video to H.264 MP4.
    """
    command = [
        "ffmpeg",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(final_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    if not final_path.exists():
        raise FileNotFoundError(
            "The converted H.264 video was not created."
        )


def download_video(
    video_url: str,
    video_id: str,
) -> Path:
    """
    Download a TikTok video and save it as MP4.
    """
    final_video_path = (
        VIDEOS_FOLDER
        / f"{video_id}.mp4"
    )

    if final_video_path.exists():
        return final_video_path

    temporary_template = str(
        VIDEOS_FOLDER
        / f"{video_id}_source.%(ext)s"
    )

    options = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": temporary_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
    }

    with YoutubeDL(options) as downloader:
        downloader.extract_info(
            video_url,
            download=True,
        )

    temporary_files = [
        path
        for path in VIDEOS_FOLDER.glob(
            f"{video_id}_source.*"
        )
        if path.is_file()
    ]

    if not temporary_files:
        raise FileNotFoundError(
            "The downloaded source video was not found."
        )

    source_video_path = temporary_files[0]

    convert_to_h264(
        source_path=source_video_path,
        final_path=final_video_path,
    )

    try:
        if (
            source_video_path.exists()
            and source_video_path != final_video_path
        ):
            source_video_path.unlink()
    except OSError:
        print(
            "Warning: Could not remove temporary "
            f"video: {source_video_path.name}"
        )

    return final_video_path