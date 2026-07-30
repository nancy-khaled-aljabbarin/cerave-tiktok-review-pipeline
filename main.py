from pathlib import Path

import pandas as pd

from src.data_collection.build_final_dataset import (
    main as build_final_dataset,
)
from src.data_collection.collect_tiktok import (
    main as collect_tiktok,
)
from src.data_collection.prepare_reviews import (
    main as prepare_reviews,
)
from src.speech_to_text.process_final_dataset import (
    process_dataset,
)


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"

RAW_FILE = DATA_FOLDER / "raw_tiktok_videos.csv"

BATCH_1_FILE = (
    DATA_FOLDER
    / "review_ready_batch1.csv"
)

BATCH_2_FILE = (
    DATA_FOLDER
    / "review_tiktok_videos_batch2.csv"
)

FINAL_FILE = (
    DATA_FOLDER
    / "cerave_reviews_final.csv"
)

TRANSCRIPTION_FILE = (
    DATA_FOLDER
    / "cerave_reviews_enriched.csv"
)

EXPECTED_SENTIMENT_COUNTS = {
    "positive": 17,
    "negative": 17,
    "neutral": 16,
}


def read_csv_file(file_path):
    """
    Read a CSV file without changing it.
    """
    try:
        return pd.read_csv(
            file_path,
            keep_default_na=False,
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
    ):
        return None


def csv_is_ready(
    file_path,
    required_columns,
):
    """
    Check that a CSV file exists and is valid.
    """
    if not file_path.exists():
        return False

    dataset = read_csv_file(file_path)

    if dataset is None or dataset.empty:
        return False

    return set(required_columns).issubset(
        dataset.columns
    )


def review_is_complete(file_path):
    """
    Check that the manual review is complete.
    """
    dataset = read_csv_file(file_path)

    if dataset is None or dataset.empty:
        return False

    required_columns = {
        "manual_keep",
        "final_sentiment",
    }

    if not required_columns.issubset(
        dataset.columns
    ):
        return False

    manual_keep = (
        dataset["manual_keep"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    final_sentiment = (
        dataset["final_sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Every row must be reviewed.
    if not manual_keep.isin(
        ["yes", "no"]
    ).all():
        return False

    accepted_rows = manual_keep.eq("yes")

    if not accepted_rows.any():
        return False

    valid_sentiments = {
        "positive",
        "negative",
        "neutral",
        "mixed",
    }

    return final_sentiment[
        accepted_rows
    ].isin(valid_sentiments).all()


def final_dataset_is_ready():
    """
    Check the final dataset columns and counts.
    """
    required_columns = {
        "video_url",
        "video_description",
        "like_count",
        "sentiment",
    }

    if not csv_is_ready(
        FINAL_FILE,
        required_columns,
    ):
        return False

    dataset = read_csv_file(FINAL_FILE)

    if len(dataset) != 50:
        return False

    sentiment = (
        dataset["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    counts = sentiment.value_counts()

    for label, expected_count in (
        EXPECTED_SENTIMENT_COUNTS.items()
    ):
        if counts.get(label, 0) != expected_count:
            return False

    return True


def transcription_file_is_safe():
    """
    Make sure an old transcription file
    will not be matched with the wrong videos.
    """
    if not TRANSCRIPTION_FILE.exists():
        return True

    dataset = read_csv_file(
        TRANSCRIPTION_FILE
    )

    if dataset is None:
        print(
            "\nThe existing transcription file "
            "could not be read."
        )
        print(
            "The file was not changed."
        )
        return False

    if dataset.empty:
        return True

    if "video_url" not in dataset.columns:
        print(
            "\nAn old transcription file was found:"
        )
        print(TRANSCRIPTION_FILE)

        print(
            "It does not contain the video_url column, "
            "so the pipeline stopped to protect "
            "the previous results."
        )

        return False

    return True


def main():
    """
    Run the project pipeline step by step.
    Completed steps are not repeated.
    """
    print("\nStarting the project pipeline...\n")

    if FINAL_FILE.exists():
        if not final_dataset_is_ready():
            print(
                "The final dataset exists, but it "
                "is incomplete or has incorrect data."
            )
            print(
                "The file was not changed."
            )
            return

        print(
            "Final dataset is ready. "
            "Skipping data collection and filtering."
        )

    else:
        if not BATCH_2_FILE.exists():
            raw_columns = {
                "video_url",
                "video_description",
                "like_count",
            }

            if RAW_FILE.exists():
                if not csv_is_ready(
                    RAW_FILE,
                    raw_columns,
                ):
                    print(
                        "The raw dataset exists, "
                        "but it is not valid."
                    )
                    print(
                        "The file was not changed."
                    )
                    return

                print(
                    "Scraping results already exist. "
                    "Skipping scraping."
                )

            else:
                print("Running TikTok scraping...")
                collect_tiktok()

                if not csv_is_ready(
                    RAW_FILE,
                    raw_columns,
                ):
                    print(
                        "Scraping did not create "
                        "a valid dataset."
                    )
                    return

            print("\nPreparing reviews...")
            prepare_reviews()

            print(
                "\nThe review file was created:"
            )
            print(BATCH_2_FILE)

            print(
                "Complete manual_keep and "
                "final_sentiment, then run "
                "main.py again."
            )
            return

        print(
            "The prepared review file already exists. "
            "It will not be created again."
        )

        if not BATCH_1_FILE.exists():
            print(
                "\nThe first reviewed batch "
                "was not found:"
            )
            print(BATCH_1_FILE)
            return

        if not review_is_complete(
            BATCH_1_FILE
        ):
            print(
                "\nThe manual review for batch 1 "
                "is not complete."
            )
            print(
                "The file was not changed."
            )
            return

        if not review_is_complete(
            BATCH_2_FILE
        ):
            print(
                "\nThe manual review for batch 2 "
                "is not complete."
            )
            print(
                "Complete manual_keep and "
                "final_sentiment, then run "
                "main.py again."
            )
            return

        print("\nBuilding the final dataset...")

        try:
            build_final_dataset()

        except (
            FileNotFoundError,
            ValueError,
        ) as error:
            print(
                "The final dataset could not "
                f"be created: {error}"
            )
            return

        if not final_dataset_is_ready():
            print(
                "The final dataset was not "
                "created correctly."
            )
            return

    if not transcription_file_is_safe():
        return

    print(
        "\nStarting video downloading "
        "and transcription..."
    )

    process_dataset()

    print(
        "\nAll available pipeline steps "
        "are completed."
    )


if __name__ == "__main__":
    main()