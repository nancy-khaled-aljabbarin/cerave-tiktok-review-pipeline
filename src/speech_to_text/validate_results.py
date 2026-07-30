from pathlib import Path

import pandas as pd

from .config import OUTPUT_CSV, PROJECT_ROOT


EXPECTED_TOTAL = 50

CERAVE_WRONG_FORMS = [
    "sarah v",
    "survey moisturizer",
    "survey cleanser",
    "serve it moisturizer",
    "serve it cleanser",
    "ceravi",
    "ceravee",
    "crb",
]


def resolve_project_path(path_value: str) -> Path:
    """
    Convert a relative CSV path into a project path.
    """
    file_path = Path(str(path_value).strip())

    if file_path.is_absolute():
        return file_path

    return PROJECT_ROOT / file_path


def contains_wrong_cerave_form(text: str) -> bool:
    """
    Check whether a known incorrect CeraVe spelling remains.
    """
    lowercase_text = str(text).lower()

    return any(
        wrong_form in lowercase_text
        for wrong_form in CERAVE_WRONG_FORMS
    )


def validate_results() -> None:
    """
    Validate downloaded files and transcription outputs.
    """

    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(
            f"Output CSV was not found: {OUTPUT_CSV}"
        )

    dataset = pd.read_csv(
        OUTPUT_CSV,
        keep_default_na=False,
    )

    required_columns = {
        "video_path",
        "audio_path",
        "raw_transcription",
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

    issues = []

    for row_index, row in dataset.iterrows():
        video_id = f"video_{row_index + 1:03d}"

        video_path = resolve_project_path(
            row["video_path"]
        )

        audio_path = resolve_project_path(
            row["audio_path"]
        )

        raw_transcription = str(
            row["raw_transcription"]
        ).strip()

        cleaned_transcription = str(
            row["transcription"]
        ).strip()

        row_issues = []

        if not video_path.exists():
            row_issues.append(
                "video file is missing"
            )

        if not audio_path.exists():
            row_issues.append(
                "audio file is missing"
            )

        if not raw_transcription:
            row_issues.append(
                "raw transcription is empty"
            )

        if not cleaned_transcription:
            row_issues.append(
                "cleaned transcription is empty"
            )

        if contains_wrong_cerave_form(
            cleaned_transcription
        ):
            row_issues.append(
                "possible CeraVe typo remains"
            )

        if row_issues:
            issues.append(
                {
                    "video_id": video_id,
                    "issues": "; ".join(row_issues),
                }
            )

    videos_folder = PROJECT_ROOT / "data" / "videos"
    audios_folder = PROJECT_ROOT / "data" / "audios"

    video_count = len(
        list(videos_folder.glob("video_*.mp4"))
    )

    audio_count = len(
        list(audios_folder.glob("video_*.mp3"))
    )

    print("\nValidation summary")
    print("------------------")
    print(f"CSV rows: {len(dataset)}")
    print(f"Video files: {video_count}")
    print(f"Audio files: {audio_count}")

    print(
        "CSV row count:",
        "PASS"
        if len(dataset) == EXPECTED_TOTAL
        else "FAIL",
    )

    print(
        "Video count:",
        "PASS"
        if video_count == EXPECTED_TOTAL
        else "FAIL",
    )

    print(
        "Audio count:",
        "PASS"
        if audio_count == EXPECTED_TOTAL
        else "FAIL",
    )

    if issues:
        print("\nRows that need review:")

        for issue in issues:
            print(
                f"- {issue['video_id']}: "
                f"{issue['issues']}"
            )
    else:
        print(
            "\nAutomatic validation: PASS"
        )

    print(
        "\nNote: Automatic validation confirms that "
        "the pipeline outputs are complete and structurally "
        "valid. It cannot guarantee word-for-word "
        "transcription accuracy."
    )


if __name__ == "__main__":
    validate_results()