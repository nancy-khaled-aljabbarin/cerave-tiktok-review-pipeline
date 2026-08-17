from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cerave_reviews_with_llm_zero_shot.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "llm_error_analysis.csv"
)


def main():
    """Export misclassified reviews for manual error analysis."""

    dataframe = pd.read_csv(RESULT_FILE)

    manual = (
        dataframe["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    predicted = (
        dataframe["llm_sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    errors = dataframe[
        manual != predicted
    ].copy()

    errors.insert(
        0,
        "review_id",
        errors.index + 1,
    )

    selected_columns = [
        "review_id",
        "video_url",
        "transcription",
        "sentiment",
        "llm_sentiment",
        "final_facial_expression",
        "facial_reliability",
    ]

    errors = errors[
        selected_columns
    ]

    errors["possible_error_reason"] = ""

    errors.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Misclassified reviews: {len(errors)}"
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()