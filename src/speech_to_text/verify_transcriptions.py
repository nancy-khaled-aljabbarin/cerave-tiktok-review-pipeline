import math
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

from .config import (
    COMPUTE_TYPE,
    DEVICE,
    OUTPUT_CSV,
    PROJECT_ROOT,
    VERIFICATION_MODEL_NAME,
)
from .text_preprocessor import preprocess_text


INPUT_FILE = OUTPUT_CSV

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "transcription_verification.csv"
)

SIMILARITY_THRESHOLD = 0.60
CONFIDENCE_THRESHOLD = 0.70
LANGUAGE_THRESHOLD = 0.80
WORD_COUNT_THRESHOLD = 0.80


def normalize_text(text: object) -> str:
    """
    Convert text to a clean lowercase form.
    """
    if pd.isna(text):
        return ""

    return " ".join(
        str(text).lower().split()
    )


def calculate_agreement(
    first_text: str,
    second_text: str,
) -> float:
    """
    Calculate similarity between two transcriptions.
    """
    first_words = normalize_text(
        first_text
    ).split()

    second_words = normalize_text(
        second_text
    ).split()

    if not first_words or not second_words:
        return 0.0

    return SequenceMatcher(
        None,
        first_words,
        second_words,
    ).ratio()


def calculate_word_count_ratio(
    first_text: str,
    second_text: str,
) -> float:
    """
    Compare the word counts of two transcriptions.
    """
    first_count = len(
        normalize_text(first_text).split()
    )

    second_count = len(
        normalize_text(second_text).split()
    )

    if first_count == 0 or second_count == 0:
        return 0.0

    return (
        min(first_count, second_count)
        / max(first_count, second_count)
    )


def get_audio_path(
    audio_path: object,
) -> Path:
    """
    Convert a stored audio path into a full path.
    """
    path = Path(
        str(audio_path).strip()
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
) -> dict:
    """
    Transcribe audio using the verification model.
    """
    segments, information = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
    )

    text_parts = []
    confidence_values = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            text_parts.append(text)

        confidence_values.append(
            math.exp(segment.avg_logprob)
        )

    transcription = " ".join(
        text_parts
    )

    if confidence_values:
        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        average_confidence = 0.0

    return {
        "text": transcription,
        "confidence": average_confidence,
        "language": information.language,
        "language_probability": (
            information.language_probability
        ),
    }


def has_useful_speech(text: str) -> bool:
    """
    Check whether the transcription contains useful speech.
    """
    words = normalize_text(text).split()

    if len(words) < 5:
        return False

    unique_word_ratio = (
        len(set(words))
        / len(words)
    )

    if unique_word_ratio < 0.30:
        return False

    return True


def save_results(
    results: list[dict],
) -> None:
    """
    Save verification progress.
    """
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def verify_transcriptions() -> None:
    """
    Verify transcriptions using a second Whisper model.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    dataset = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    required_columns = {
        "video_path",
        "audio_path",
        "transcription",
    }

    missing_columns = required_columns - set(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    print(
        "Loading verification Whisper model: "
        f"{VERIFICATION_MODEL_NAME}"
    )

    verification_model = WhisperModel(
        VERIFICATION_MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    results = []
    total_videos = len(dataset)

    for index, row in dataset.iterrows():
        current_number = index + 1

        print(
            f"[{current_number}/{total_videos}] "
            "Verifying..."
        )

        video_id = Path(
            str(row["video_path"])
        ).stem

        original_transcription = str(
            row["transcription"]
        ).strip()

        audio_path = get_audio_path(
            row["audio_path"]
        )

        second_transcription = ""
        agreement_score = 0.0
        model_confidence = 0.0
        detected_language = ""
        language_probability = 0.0
        word_count_ratio = 0.0
        verification_status = "needs_review"
        verification_error = ""

        try:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            model_result = transcribe_audio(
                verification_model,
                audio_path,
            )

            second_transcription = preprocess_text(
                model_result["text"]
            )

            model_confidence = float(
                model_result["confidence"]
            )

            detected_language = str(
                model_result["language"]
            )

            language_probability = float(
                model_result[
                    "language_probability"
                ]
            )

            agreement_score = calculate_agreement(
                original_transcription,
                second_transcription,
            )

            word_count_ratio = (
                calculate_word_count_ratio(
                    original_transcription,
                    second_transcription,
                )
            )

            if not has_useful_speech(
                second_transcription
            ):
                verification_status = "no_speech"

            elif (
                agreement_score
                >= SIMILARITY_THRESHOLD
                and model_confidence
                >= CONFIDENCE_THRESHOLD
                and detected_language == "en"
                and language_probability
                >= LANGUAGE_THRESHOLD
                and word_count_ratio
                >= WORD_COUNT_THRESHOLD
            ):
                verification_status = "verified_auto"

            else:
                verification_status = "needs_review"

        except Exception as error:
            verification_status = (
                "verification_failed"
            )

            verification_error = (
                f"{type(error).__name__}: {error}"
            )

            print(
                f"Error in {video_id}: "
                f"{verification_error}"
            )

        if verification_status == "verified_auto":
            final_transcription = (
                second_transcription
            )
        else:
            final_transcription = (
                original_transcription
            )

        results.append(
            {
                "video_id": video_id,
                "original_transcription": (
                    original_transcription
                ),
                "second_transcription": (
                    second_transcription
                ),
                "agreement_score": round(
                    agreement_score,
                    4,
                ),
                "model_confidence": round(
                    model_confidence,
                    4,
                ),
                "detected_language": (
                    detected_language
                ),
                "language_probability": round(
                    language_probability,
                    4,
                ),
                "word_count_ratio": round(
                    word_count_ratio,
                    4,
                ),
                "verification_status": (
                    verification_status
                ),
                "final_transcription": (
                    final_transcription
                ),
                "verification_error": (
                    verification_error
                ),
            }
        )

        save_results(results)

    results_dataset = pd.DataFrame(
        results
    )

    status_counts = (
        results_dataset[
            "verification_status"
        ].value_counts()
    )

    verified_count = int(
        status_counts.get(
            "verified_auto",
            0,
        )
    )

    review_count = int(
        status_counts.get(
            "needs_review",
            0,
        )
    )

    no_speech_count = int(
        status_counts.get(
            "no_speech",
            0,
        )
    )

    failed_count = int(
        status_counts.get(
            "verification_failed",
            0,
        )
    )

    print("\nVerification completed.")
    print(f"Total videos: {total_videos}")
    print(
        f"Automatically verified: {verified_count}"
    )
    print(f"Need review: {review_count}")
    print(f"No speech: {no_speech_count}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    verify_transcriptions()