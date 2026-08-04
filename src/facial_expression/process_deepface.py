from pathlib import Path

import pandas as pd

from .config import DATA_DIR, OUTPUT_CSV, PROJECT_ROOT
from .deepface_analyzer import analyze_video_deepface


DEEPFACE_OUTPUT_CSV = (
    DATA_DIR
    / "cerave_reviews_with_two_models.csv"
)

RESULT_COLUMNS = {
    "deepface_expression": "",
    "deepface_confidence": 0.0,
    "deepface_status": "",
    "deepface_frames_analyzed": 0,
    "deepface_agreement": "",
    "deepface_model": "",
    "deepface_error": "",
}


def resolve_video_path(video_path):
    """Convert the CSV video path into a complete path."""

    path = Path(str(video_path))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def process_deepface_dataset(limit=None):
    """Analyze the videos using DeepFace."""

    if DEEPFACE_OUTPUT_CSV.exists():
        dataframe = pd.read_csv(
            DEEPFACE_OUTPUT_CSV
        )
    else:
        dataframe = pd.read_csv(
            OUTPUT_CSV
        )

    if "video_path" not in dataframe.columns:
        raise ValueError(
            "The input CSV does not contain video_path."
        )

    for column, default_value in RESULT_COLUMNS.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

    pending_indices = dataframe.index[
        dataframe["deepface_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ].tolist()

    if limit is not None:
        pending_indices = pending_indices[:limit]

    print(
        f"Videos selected: {len(pending_indices)}"
    )

    for number, index in enumerate(
        pending_indices,
        start=1,
    ):
        video_path = resolve_video_path(
            dataframe.at[index, "video_path"]
        )

        print(
            f"\n[{number}/{len(pending_indices)}] "
            f"Processing: {video_path.name}"
        )

        try:
            result = analyze_video_deepface(
                video_path
            )

            for column, value in result.items():
                dataframe.at[index, column] = value

            dataframe.at[
                index,
                "deepface_error",
            ] = ""

            print(
                "Result:",
                result["deepface_expression"],
            )

            print(
                "Confidence:",
                f"{result['deepface_confidence']:.2%}",
            )

            print(
                "Agreement:",
                result["deepface_agreement"],
            )

        except Exception as error:
            dataframe.at[
                index,
                "deepface_status",
            ] = "failed"

            dataframe.at[
                index,
                "deepface_error",
            ] = str(error)

            print(
                "Failed:",
                error,
            )

        dataframe.to_csv(
            DEEPFACE_OUTPUT_CSV,
            index=False,
        )

    print(
        "\nDeepFace processing finished."
    )

    print(
        "Output:",
        DEEPFACE_OUTPUT_CSV,
    )

    print("\nStatus summary:")

    print(
        dataframe[
            "deepface_status"
        ].value_counts(
            dropna=False
        )
    )


def main():
    """Process all pending videos with DeepFace."""

    process_deepface_dataset()

