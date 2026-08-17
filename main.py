import subprocess

import pandas as pd

from config import (
    BATCH_1_FILE,
    BATCH_2_FILE,
    CONDA_EXE,
    EXPECTED_SENTIMENT_COUNTS,
    FACIAL_ENV_NAME,
    FINAL_FILE,
    LLM_ENV_NAME,
    PROJECT_ROOT,
    RAW_FILE,
    TRANSCRIPTION_FILE,
)
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


def read_csv_file(file_path):
    """Read a CSV file safely."""

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
    """Check that a CSV file exists and is valid."""

    if not file_path.exists():
        return False

    dataset = read_csv_file(file_path)

    if dataset is None or dataset.empty:
        return False

    return set(required_columns).issubset(
        dataset.columns
    )


def review_is_complete(file_path):
    """Check that the manual review is complete."""

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
    """Check the final dataset structure."""

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

    sentiment_counts = (
        dataset["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
    )

    for label, expected_count in (
        EXPECTED_SENTIMENT_COUNTS.items()
    ):
        if (
            sentiment_counts.get(label, 0)
            != expected_count
        ):
            return False

    return True


def transcription_file_is_safe():
    """Protect previous transcription results."""

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
        return False

    if dataset.empty:
        return True

    if "video_url" not in dataset.columns:
        print(
            "\nThe existing transcription file "
            "does not contain video_url."
        )
        print(
            "The pipeline stopped to protect "
            "the previous results."
        )
        return False

    return True


def run_conda_stage(
    module_name,
    function_name,
    env_name,
    function_arguments="",
):
    """Run one pipeline stage in the selected Conda environment."""

    if not CONDA_EXE.exists():
        raise FileNotFoundError(
            f"Conda was not found: {CONDA_EXE}"
        )

    function_call = (
        f"{function_name}({function_arguments})"
    )

    python_code = (
        f"from {module_name} import "
        f"{function_name}; "
        f"{function_call}"
    )

    subprocess.run(
        [
            str(CONDA_EXE),
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-c",
            python_code,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def prepare_final_dataset():
    """Create the reviewed final dataset when needed."""

    if FINAL_FILE.exists():
        if not final_dataset_is_ready():
            print(
                "The final dataset exists, but "
                "its data is not valid."
            )
            return False

        print(
            "Final dataset is ready. "
            "Skipping data collection."
        )
        return True

    raw_columns = {
        "video_url",
        "video_description",
        "like_count",
    }

    if not BATCH_2_FILE.exists():
        if RAW_FILE.exists():
            if not csv_is_ready(
                RAW_FILE,
                raw_columns,
            ):
                print(
                    "The raw dataset is not valid."
                )
                return False
        else:
            print(
                "Running TikTok scraping..."
            )
            collect_tiktok()

        print(
            "\nPreparing reviews..."
        )
        prepare_reviews()

        print(
            "\nComplete the manual review, "
            "then run main.py again."
        )
        return False

    if not BATCH_1_FILE.exists():
        print(
            "The first reviewed batch "
            "was not found."
        )
        return False

    if not review_is_complete(
        BATCH_1_FILE
    ):
        print(
            "The manual review for batch 1 "
            "is not complete."
        )
        return False

    if not review_is_complete(
        BATCH_2_FILE
    ):
        print(
            "The manual review for batch 2 "
            "is not complete."
        )
        return False

    print(
        "\nBuilding the final dataset..."
    )
    build_final_dataset()

    if not final_dataset_is_ready():
        print(
            "The final dataset was not "
            "created correctly."
        )
        return False

    return True


def main():
    """Run the complete project pipeline."""

    print(
        "\nStarting the project pipeline...\n"
    )

    if not prepare_final_dataset():
        return

    if not transcription_file_is_safe():
        return

    print(
        "\nStarting video downloading "
        "and transcription..."
    )
    process_dataset()

    print(
        "\nStarting facial-expression analysis..."
    )
    run_conda_stage(
        module_name=(
            "src.facial_expression."
            "process_facial_expressions"
        ),
        function_name="main",
        env_name=FACIAL_ENV_NAME,
    )

    print(
        "\nStarting DeepFace analysis..."
    )
    run_conda_stage(
        module_name=(
            "src.facial_expression."
            "process_deepface"
        ),
        function_name="main",
        env_name=FACIAL_ENV_NAME,
    )

    print(
        "\nStarting speech detection..."
    )
    run_conda_stage(
        module_name="src.speech_detector",
        function_name="main",
        env_name=FACIAL_ENV_NAME,
    )

    print(
        "\nStarting LLM sentiment analysis..."
    )
    run_conda_stage(
        module_name=(
            "src.llm_analysis."
            "process_reviews"
        ),
        function_name="process_reviews",
        env_name=LLM_ENV_NAME,
        function_arguments=(
            "prompt_type='zero_shot'"
        ),
    )

    print(
        "\nAll pipeline stages "
        "are completed."
    )


if __name__ == "__main__":
    main()