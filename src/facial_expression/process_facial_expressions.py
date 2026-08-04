from pathlib import Path

import pandas as pd

from .config import (
    INPUT_CSV,
    OUTPUT_CSV,
    PROJECT_ROOT,
)
from .model_loader import load_models
from .video_analyzer import analyze_video


# Columns produced by the facial-expression stage.
RESULT_COLUMNS = {
    "facial_expression": "",
    "facial_expression_confidence": 0.0,
    "facial_expression_status": "",
    "face_frames_analyzed": 0,
    "valid_windows": 0,
    "window_agreement": "",
    "confidence_margin": 0.0,
    "facial_expression_model": "",
    "facial_expression_error": "",
}


# These statuses do not need to be processed again.
COMPLETED_STATUSES = {
    "success",
    "uncertain",
    "no_face",
}


def add_result_columns(dataframe):
    """Add missing facial-expression result columns."""

    for column, default_value in RESULT_COLUMNS.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

    return dataframe


def restore_previous_results(dataframe):
    """Restore results saved during an earlier run."""

    if not OUTPUT_CSV.exists():
        return dataframe

    previous_data = pd.read_csv(OUTPUT_CSV)

    if "video_url" not in previous_data.columns:
        return dataframe

    previous_data = previous_data.drop_duplicates(
        subset="video_url",
        keep="last",
    )

    previous_data = previous_data.set_index(
        "video_url"
    )

    for index, row in dataframe.iterrows():
        video_url = row["video_url"]

        if video_url not in previous_data.index:
            continue

        previous_row = previous_data.loc[
            video_url
        ]

        for column in RESULT_COLUMNS:
            if column not in previous_data.columns:
                continue

            value = previous_row[column]

            if pd.notna(value):
                dataframe.at[index, column] = value

    return dataframe


def resolve_video_path(path_value):
    """Convert a CSV video path into a complete project path."""

    if pd.isna(path_value):
        return None

    path_text = str(path_value).strip()

    if not path_text:
        return None

    video_path = Path(path_text)

    if video_path.is_absolute():
        return video_path

    return PROJECT_ROOT / video_path


def save_results(dataframe):
    """Save progress after every processed video."""

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def get_pending_indices(dataframe):
    """Return the rows that still need facial analysis."""

    pending_indices = []

    for index, row in dataframe.iterrows():
        status = str(
            row["facial_expression_status"]
        ).strip().lower()

        if status in COMPLETED_STATUSES:
            continue

        pending_indices.append(index)

    return pending_indices


def store_result(dataframe, index, result):
    """Store one video's analysis result in the dataset."""

    for column, value in result.items():
        if column in RESULT_COLUMNS:
            dataframe.at[index, column] = value

    dataframe.at[
        index,
        "facial_expression_error",
    ] = ""


def store_failure(dataframe, index, error):
    """Store an error without stopping the complete pipeline."""

    dataframe.at[
        index,
        "facial_expression_status",
    ] = "failed"

    dataframe.at[
        index,
        "facial_expression_error",
    ] = str(error)


def process_dataset(limit=None):
    """Analyze facial expressions for pending videos."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {INPUT_CSV}"
        )

    dataframe = pd.read_csv(INPUT_CSV)

    if "video_path" not in dataframe.columns:
        raise ValueError(
            "The input dataset does not contain "
            "the required 'video_path' column."
        )

    if "video_url" not in dataframe.columns:
        raise ValueError(
            "The input dataset does not contain "
            "the required 'video_url' column."
        )

    dataframe = add_result_columns(dataframe)
    dataframe = restore_previous_results(dataframe)

    pending_indices = get_pending_indices(
        dataframe
    )

    if limit is not None:
        if limit < 1:
            raise ValueError(
                "The processing limit must be at least 1."
            )

        pending_indices = pending_indices[:limit]

    if not pending_indices:
        save_results(dataframe)

        print(
            "All facial-expression results "
            "are already completed."
        )

        return dataframe

    print("Loading facial-expression models...")

    static_model, dynamic_model, device = load_models()

    total_pending = len(pending_indices)

    for position, index in enumerate(
        pending_indices,
        start=1,
    ):
        video_path = resolve_video_path(
            dataframe.at[index, "video_path"]
        )

        video_name = (
            video_path.name
            if video_path is not None
            else "missing_video_path"
        )

        print(
            f"\n[{position}/{total_pending}] "
            f"Processing: {video_name}"
        )

        try:
            if video_path is None:
                raise ValueError(
                    "The video path is missing."
                )

            result = analyze_video(
                video_path=video_path,
                static_model=static_model,
                dynamic_model=dynamic_model,
                device=device,
            )

            store_result(
                dataframe=dataframe,
                index=index,
                result=result,
            )

            print(
                "Result:",
                result["facial_expression"],
                f"({result['facial_expression_status']})",
            )

            print(
                "Confidence:",
                f"{result['facial_expression_confidence']:.2%}",
            )

            print(
                "Window agreement:",
                result["window_agreement"],
            )

            print(
                "Confidence margin:",
                f"{result['confidence_margin']:.2%}",
            )

        except Exception as error:
            store_failure(
                dataframe=dataframe,
                index=index,
                error=error,
            )

            print("Failed:", error)

        # Save after each video so progress is not lost.
        save_results(dataframe)

    print(
        "\nFacial-expression processing finished."
    )

    print("Output:", OUTPUT_CSV)

    status_counts = dataframe[
        "facial_expression_status"
    ].value_counts(
        dropna=False
    )

    print("\nStatus summary:")
    print(status_counts.to_string())

    return dataframe


def main():
    """Process all pending videos."""

    process_dataset()


if __name__ == "__main__":
    main()