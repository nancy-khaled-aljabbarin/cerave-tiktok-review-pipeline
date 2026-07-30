from pathlib import Path

import pandas as pd

from .config import OUTPUT_CSV, PROJECT_ROOT


INPUT_FILE = OUTPUT_CSV

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "transcription_evaluation.csv"
)


def get_video_id(video_path: object) -> str:
    """
    Extract the video ID from the video path.
    """
    return Path(
        str(video_path)
    ).stem


def prepare_evaluation_file() -> None:
    """
    Create a CSV file for manual transcription evaluation.
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

    evaluation_dataset = pd.DataFrame()

    evaluation_dataset["video_id"] = (
        dataset["video_path"].apply(
            get_video_id
        )
    )

    evaluation_dataset[
        "generated_transcription"
    ] = dataset["transcription"]

    evaluation_dataset[
        "human_reference"
    ] = dataset["transcription"]

    evaluation_dataset[
        "review_status"
    ] = "not_reviewed"

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Evaluation file created successfully."
    )
    print(
        f"Number of videos: {len(evaluation_dataset)}"
    )
    print(f"Saved in: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_evaluation_file()