from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
import pandas as pd


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "silero_vad.onnx"

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "cerave_reviews_with_two_models.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "speech_check.csv"
)


# Model settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SPEECH_THRESHOLD = 0.50


MIN_SPEECH_SECONDS = 0.25


# Process all videos
LIMIT = None


def extract_audio(video_path):
    """Extract mono 16 kHz audio from one video."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        return None

    return np.frombuffer(
        result.stdout,
        dtype=np.float32,
    )


def detect_speech(session, audio):
    """Check whether the audio contains human speech."""

    state = np.zeros(
        (2, 1, 128),
        dtype=np.float32,
    )

    speech_duration = 0.0
    current_speech = 0.0
    longest_speech = 0.0
    max_probability = 0.0

    for start in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[start:start + CHUNK_SIZE]

        real_size = len(chunk)

        if real_size == 0:
            continue

        chunk_duration = real_size / SAMPLE_RATE

        if real_size < CHUNK_SIZE:
            chunk = np.pad(
                chunk,
                (0, CHUNK_SIZE - real_size),
            )

        chunk = chunk.reshape(1, -1).astype(
            np.float32
        )

        output, state = session.run(
            None,
            {
                "input": chunk,
                "state": state,
                "sr": np.array(
                    SAMPLE_RATE,
                    dtype=np.int64,
                ),
            },
        )

        probability = float(output[0][0])

        max_probability = max(
            max_probability,
            probability,
        )

        if probability >= SPEECH_THRESHOLD:
            speech_duration += chunk_duration
            current_speech += chunk_duration

            longest_speech = max(
                longest_speech,
                current_speech,
            )
        else:
            current_speech = 0.0

    audio_duration = len(audio) / SAMPLE_RATE

    speech_ratio = (
        speech_duration / audio_duration
        if audio_duration > 0
        else 0.0
    )

    speech_present = (
        longest_speech >= MIN_SPEECH_SECONDS
    )

    return {
        "speech_present": (
            "yes" if speech_present else "no"
        ),
        "speech_duration_seconds": round(
            speech_duration,
            2,
        ),
        "speech_ratio": round(
            speech_ratio,
            4,
        ),
        "max_speech_probability": round(
            max_probability,
            4,
        ),
    }


def main():
    """Check all selected videos automatically."""

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    data = pd.read_csv(
        INPUT_CSV,
        usecols=["video_path"],
    )

    if LIMIT is not None:
        data = data.head(LIMIT)

    results = []

    total = len(data)

    print(f"Videos selected: {total}")

    for number, video_value in enumerate(
        data["video_path"],
        start=1,
    ):
        video_path = (
            PROJECT_ROOT / Path(str(video_value))
        )

        print(
            f"\n[{number}/{total}] "
            f"Checking: {video_path.name}"
        )

        row = {
            "video_path": str(video_value),
            "audio_present": "no",
            "speech_present": "no",
            "audio_duration_seconds": 0.0,
            "speech_duration_seconds": 0.0,
            "speech_ratio": 0.0,
            "max_speech_probability": 0.0,
            "vad_status": "",
            "vad_model": "deepghs/silero-vad-onnx",
        }

        if not video_path.exists():
            row["vad_status"] = "video_not_found"
            results.append(row)

            print("Status: video not found")
            continue

        audio = extract_audio(video_path)

        if audio is None:
            row["vad_status"] = (
                "audio_extraction_failed"
            )
            results.append(row)

            print("Status: audio extraction failed")
            continue

        if len(audio) == 0:
            row["vad_status"] = "no_audio"
            results.append(row)

            print("Audio: no")
            print("Speech: no")
            continue

        row["audio_present"] = "yes"

        row["audio_duration_seconds"] = round(
            len(audio) / SAMPLE_RATE,
            2,
        )

        speech_result = detect_speech(
            session,
            audio,
        )

        row.update(speech_result)
        row["vad_status"] = "success"

        results.append(row)

        print("Audio:", row["audio_present"])
        print("Speech:", row["speech_present"])
        print(
            "Speech duration:",
            row["speech_duration_seconds"],
        )
        print("Speech ratio:", row["speech_ratio"])
        print("Status:", row["vad_status"])

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nSpeech detection finished.")
    print("Output:", OUTPUT_CSV)

    print("\nSpeech summary:")
    print(
        output["speech_present"]
        .value_counts()
        .to_string()
    )

    print("\nStatus summary:")
    print(
        output["vad_status"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()