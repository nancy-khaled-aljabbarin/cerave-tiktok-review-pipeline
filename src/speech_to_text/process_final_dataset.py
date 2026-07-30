import pandas as pd

from .audio_extractor import extract_audio
from .config import (
    INPUT_CSV,
    OUTPUT_CSV,
    PROJECT_ROOT,
    TEST_LIMIT,
)
from .text_preprocessor import preprocess_text
from .transcriber import transcribe_media
from .video_downloader import download_video


OUTPUT_TEXT_COLUMNS = [
    "video_path",
    "audio_path",
    "raw_transcription",
    "transcription",
    "detected_language",
    "transcription_status",
    "transcription_error",
]

OUTPUT_FLOAT_COLUMNS = [
    "language_probability",
    "audio_duration_seconds",
]


def create_output_dataset(
    input_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the output dataset.
    """
    columns_to_keep = [
        "video_url",
        "video_description",
        "like_count",
    ]

    if "sentiment" in input_dataset.columns:
        columns_to_keep.append("sentiment")

    output_dataset = input_dataset[
        columns_to_keep
    ].copy()

    for column in OUTPUT_TEXT_COLUMNS:
        output_dataset[column] = ""

    for column in OUTPUT_FLOAT_COLUMNS:
        output_dataset[column] = 0.0

    return output_dataset


def load_previous_results(
    input_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load previous results so completed videos
    are not processed again.
    """
    output_dataset = create_output_dataset(
        input_dataset
    )

    if not OUTPUT_CSV.exists():
        print(
            "No previous results were found. "
            "Starting from the beginning."
        )
        return output_dataset

    try:
        saved_dataset = pd.read_csv(
            OUTPUT_CSV,
            keep_default_na=False,
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
    ) as error:
        raise RuntimeError(
            "The previous output file could not be read. "
            "It was not changed."
        ) from error

    if saved_dataset.empty:
        return output_dataset

    main_columns = {
        "video_url",
        "video_description",
        "like_count",
        "sentiment",
    }

    # Keep any extra columns from the saved file.
    for column in saved_dataset.columns:
        if column not in output_dataset.columns:
            output_dataset[column] = ""

    columns_to_restore = [
        column
        for column in saved_dataset.columns
        if column not in main_columns
    ]

    if "video_url" in saved_dataset.columns:
        saved_dataset["video_url"] = (
            saved_dataset["video_url"]
            .astype(str)
            .str.strip()
        )

        saved_dataset.drop_duplicates(
            subset=["video_url"],
            keep="last",
            inplace=True,
        )

        saved_dataset.set_index(
            "video_url",
            inplace=True,
        )

        for row_index in output_dataset.index:
            video_url = str(
                output_dataset.at[
                    row_index,
                    "video_url",
                ]
            ).strip()

            if video_url not in saved_dataset.index:
                continue

            for column in columns_to_restore:
                if column in saved_dataset.columns:
                    output_dataset.at[
                        row_index,
                        column,
                    ] = saved_dataset.at[
                        video_url,
                        column,
                    ]

    else:
        # The old output file did not include video_url,
        # so the results are restored by row order.
        rows_to_restore = min(
            len(output_dataset),
            len(saved_dataset),
        )

        for column in columns_to_restore:
            if column in saved_dataset.columns:
                output_dataset.loc[
                    :rows_to_restore - 1,
                    column,
                ] = saved_dataset.loc[
                    :rows_to_restore - 1,
                    column,
                ].to_numpy()

    print(
        "Previous results were found. "
        "Continuing from the saved progress."
    )

    return output_dataset


def get_relative_path(file_path) -> str:
    """
    Return the path relative to the project folder.
    """
    try:
        return str(
            file_path.relative_to(PROJECT_ROOT)
        )

    except ValueError:
        return str(file_path)


def save_dataset(
    dataset: pd.DataFrame,
) -> None:
    """
    Save the current results.
    """
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def process_dataset() -> None:
    """
    Download and transcribe the videos.
    Completed videos are skipped.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV was not found: {INPUT_CSV}"
        )

    input_dataset = pd.read_csv(
        INPUT_CSV,
        keep_default_na=False,
    )

    required_columns = {
        "video_url",
        "video_description",
        "like_count",
    }

    missing_columns = required_columns - set(
        input_dataset.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    input_dataset = input_dataset.head(
        TEST_LIMIT
    ).reset_index(drop=True)

    output_dataset = load_previous_results(
        input_dataset
    )

    total_videos = len(input_dataset)

    for row_index in input_dataset.index:
        position = row_index + 1
        video_id = f"video_{position:03d}"

        current_status = str(
            output_dataset.at[
                row_index,
                "transcription_status",
            ]
        ).strip().lower()

        if current_status == "success":
            print(
                f"\n[{position}/{total_videos}] "
                f"{video_id} is already completed. "
                "Skipping."
            )
            continue

        video_url = str(
            input_dataset.at[
                row_index,
                "video_url",
            ]
        ).strip()

        print(
            f"\n[{position}/{total_videos}] "
            f"Processing {video_id}..."
        )

        try:
            if not video_url:
                raise ValueError(
                    "The video URL is empty."
                )

            video_path = download_video(
                video_url=video_url,
                video_id=video_id,
            )

            audio_path = extract_audio(
                video_path=video_path,
                video_id=video_id,
            )

            output_dataset.at[
                row_index,
                "video_path",
            ] = get_relative_path(video_path)

            output_dataset.at[
                row_index,
                "audio_path",
            ] = get_relative_path(audio_path)

            transcription_result = transcribe_media(
                audio_path
            )

            raw_transcription = str(
                transcription_result.get(
                    "transcript",
                    "",
                )
            ).strip()

            cleaned_transcription = preprocess_text(
                raw_transcription
            )

            output_dataset.at[
                row_index,
                "raw_transcription",
            ] = raw_transcription

            output_dataset.at[
                row_index,
                "transcription",
            ] = cleaned_transcription

            output_dataset.at[
                row_index,
                "detected_language",
            ] = str(
                transcription_result.get(
                    "detected_language",
                    "",
                )
            )

            output_dataset.at[
                row_index,
                "language_probability",
            ] = float(
                transcription_result.get(
                    "language_probability",
                    0.0,
                )
                or 0.0
            )

            output_dataset.at[
                row_index,
                "audio_duration_seconds",
            ] = float(
                transcription_result.get(
                    "audio_duration_seconds",
                    0.0,
                )
                or 0.0
            )

            output_dataset.at[
                row_index,
                "transcription_status",
            ] = "success"

            output_dataset.at[
                row_index,
                "transcription_error",
            ] = ""

            print(
                "Raw transcription: "
                f"{raw_transcription[:150]}"
            )

            print(
                "Cleaned transcription: "
                f"{cleaned_transcription[:150]}"
            )

        except Exception as error:
            output_dataset.at[
                row_index,
                "transcription_status",
            ] = "failed"

            output_dataset.at[
                row_index,
                "transcription_error",
            ] = (
                f"{type(error).__name__}: {error}"
            )

            print(
                "Failed: "
                f"{type(error).__name__}: {error}"
            )

        finally:
            # Save after every video so the progress is not lost.
            save_dataset(output_dataset)

    successful = int(
        (
            output_dataset[
                "transcription_status"
            ]
            .astype(str)
            .str.lower()
            == "success"
        ).sum()
    )

    failed = int(
        (
            output_dataset[
                "transcription_status"
            ]
            .astype(str)
            .str.lower()
            == "failed"
        ).sum()
    )

    remaining = total_videos - successful - failed

    print("\nProcessing completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Remaining: {remaining}")
    print(f"Output file: {OUTPUT_CSV}")


if __name__ == "__main__":
    process_dataset()