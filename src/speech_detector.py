from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
import pandas as pd


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "silero_vad.onnx"
)

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

LIMIT = None


RESULT_COLUMNS = {
    "audio_present": "no",
    "speech_present": "no",
    "audio_duration_seconds": 0.0,
    "speech_duration_seconds": 0.0,
    "speech_ratio": 0.0,
    "max_speech_probability": 0.0,
    "vad_status": "",
    "vad_model": "deepghs/silero-vad-onnx",
}


COMPLETED_STATUSES = {
    "success",
    "no_audio",
}


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

    for start in range(
        0,
        len(audio),
        CHUNK_SIZE,
    ):
        chunk = audio[
            start:start + CHUNK_SIZE
        ]

        real_size = len(chunk)

        if real_size == 0:
            continue

        chunk_duration = (
            real_size / SAMPLE_RATE
        )

        if real_size < CHUNK_SIZE:
            chunk = np.pad(
                chunk,
                (0, CHUNK_SIZE - real_size),
            )

        chunk = (
            chunk.reshape(1, -1)
            .astype(np.float32)
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

        probability = float(
            output[0][0]
        )

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

    audio_duration = (
        len(audio) / SAMPLE_RATE
    )

    speech_ratio = (
        speech_duration / audio_duration
        if audio_duration > 0
        else 0.0
    )

    speech_present = (
        longest_speech
        >= MIN_SPEECH_SECONDS
    )

    return {
        "speech_present": (
            "yes"
            if speech_present
            else "no"
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


def add_result_columns(dataframe):
    """Add missing VAD result columns."""

    for column, default_value in (
        RESULT_COLUMNS.items()
    ):
        if column not in dataframe.columns:
            dataframe[column] = default_value

    return dataframe


def restore_previous_results(dataframe):
    """Restore saved VAD results from an earlier run."""

    if not OUTPUT_CSV.exists():
        return dataframe

    previous = pd.read_csv(
        OUTPUT_CSV,
        keep_default_na=False,
    )

    if "video_path" not in previous.columns:
        return dataframe

    previous = previous.drop_duplicates(
        subset="video_path",
        keep="last",
    ).set_index("video_path")

    for index, row in dataframe.iterrows():
        video_path = str(
            row["video_path"]
        ).strip()

        if video_path not in previous.index:
            continue

        previous_row = previous.loc[
            video_path
        ]

        for column in RESULT_COLUMNS:
            if column not in previous.columns:
                continue

            value = previous_row[column]

            if pd.notna(value):
                dataframe.at[
                    index,
                    column,
                ] = value

    return dataframe


def get_pending_indices(dataframe):
    """Return rows that still need VAD processing."""

    pending_indices = []

    for index, row in dataframe.iterrows():
        status = str(
            row["vad_status"]
        ).strip().lower()

        if status in COMPLETED_STATUSES:
            continue

        pending_indices.append(index)

    return pending_indices


def save_results(dataframe):
    """Save VAD progress."""

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    """Check pending videos for speech."""

    data = pd.read_csv(
        INPUT_CSV,
        usecols=["video_path"],
        keep_default_na=False,
    )

    if LIMIT is not None:
        data = data.head(
            LIMIT
        ).reset_index(drop=True)

    data = add_result_columns(data)
    data = restore_previous_results(data)

    pending_indices = get_pending_indices(
        data
    )

    print(
        f"Videos selected: "
        f"{len(pending_indices)}"
    )

    if not pending_indices:
        save_results(data)

        print(
            "All speech-detection results "
            "are already completed."
        )

        return data

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    total = len(pending_indices)

    for position, index in enumerate(
        pending_indices,
        start=1,
    ):
        video_value = str(
            data.at[
                index,
                "video_path",
            ]
        ).strip()

        video_path = (
            PROJECT_ROOT
            / Path(video_value)
        )

        print(
            f"\n[{position}/{total}] "
            f"Checking: {video_path.name}"
        )

        data.at[
            index,
            "audio_present",
        ] = "no"

        data.at[
            index,
            "speech_present",
        ] = "no"

        data.at[
            index,
            "audio_duration_seconds",
        ] = 0.0

        data.at[
            index,
            "speech_duration_seconds",
        ] = 0.0

        data.at[
            index,
            "speech_ratio",
        ] = 0.0

        data.at[
            index,
            "max_speech_probability",
        ] = 0.0

        data.at[
            index,
            "vad_model",
        ] = "deepghs/silero-vad-onnx"

        if not video_path.exists():
            data.at[
                index,
                "vad_status",
            ] = "video_not_found"

            save_results(data)

            print(
                "Status: video not found"
            )
            continue

        audio = extract_audio(
            video_path
        )

        if audio is None:
            data.at[
                index,
                "vad_status",
            ] = (
                "audio_extraction_failed"
            )

            save_results(data)

            print(
                "Status: "
                "audio extraction failed"
            )
            continue

        if len(audio) == 0:
            data.at[
                index,
                "vad_status",
            ] = "no_audio"

            save_results(data)

            print("Audio: no")
            print("Speech: no")
            continue

        data.at[
            index,
            "audio_present",
        ] = "yes"

        data.at[
            index,
            "audio_duration_seconds",
        ] = round(
            len(audio) / SAMPLE_RATE,
            2,
        )

        speech_result = detect_speech(
            session,
            audio,
        )

        for column, value in (
            speech_result.items()
        ):
            data.at[
                index,
                column,
            ] = value

        data.at[
            index,
            "vad_status",
        ] = "success"

        save_results(data)

        print(
            "Audio:",
            data.at[
                index,
                "audio_present",
            ],
        )
        print(
            "Speech:",
            data.at[
                index,
                "speech_present",
            ],
        )
        print(
            "Speech duration:",
            data.at[
                index,
                "speech_duration_seconds",
            ],
        )
        print(
            "Speech ratio:",
            data.at[
                index,
                "speech_ratio",
            ],
        )
        print(
            "Status:",
            data.at[
                index,
                "vad_status",
            ],
        )

    print(
        "\nSpeech detection finished."
    )
    print(
        "Output:",
        OUTPUT_CSV,
    )

    print(
        "\nSpeech summary:"
    )
    print(
        data["speech_present"]
        .value_counts()
        .to_string()
    )

    print(
        "\nStatus summary:"
    )
    print(
        data["vad_status"]
        .value_counts()
        .to_string()
    )

    return data


if __name__ == "__main__":
    main()