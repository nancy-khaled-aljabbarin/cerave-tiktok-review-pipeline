from pathlib import Path

import pandas as pd

from .llm_runner import load_model, classify_sentiment
from .prompt_builder import (
    build_few_shot_prompt,
    build_text_only_prompt,
    build_zero_shot_prompt,
    build_zero_shot_prompt_v3,
    build_zero_shot_prompt_v4,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cerave_reviews_with_two_models.csv"
)

LLM_COLUMN = "llm_sentiment"


def get_output_file(prompt_type):
    """Return the output file for the selected prompt type."""

    if prompt_type == "zero_shot":
        return (
            PROJECT_ROOT
            / "data"
            / "cerave_reviews_with_llm_zero_shot.csv"
        )

    if prompt_type == "zero_shot_v3":
        return (
            PROJECT_ROOT
            / "data"
            / "cerave_reviews_with_llm_v3.csv"
        )

    if prompt_type == "zero_shot_v4":
        return (
            PROJECT_ROOT
            / "data"
            / "cerave_reviews_with_llm_v4.csv"
        )

    if prompt_type == "few_shot":
        return (
            PROJECT_ROOT
            / "data"
            / "cerave_reviews_with_llm_few_shot.csv"
        )

    if prompt_type == "text_only":
        return (
            PROJECT_ROOT
            / "data"
            / "cerave_reviews_with_llm_text_only.csv"
        )

    raise ValueError(
        "prompt_type must be "
        "'zero_shot', 'zero_shot_v3', "
        "'zero_shot_v4', 'few_shot', "
        "or 'text_only'."
    )


def build_prompt(
    prompt_type,
    transcription,
    facial_expression,
    facial_reliability,
):
    """Build the selected prompt type."""

    if prompt_type == "zero_shot":
        return build_zero_shot_prompt(
            transcription=transcription,
            facial_expression=facial_expression,
            facial_reliability=facial_reliability,
        )

    if prompt_type == "zero_shot_v3":
        return build_zero_shot_prompt_v3(
            transcription=transcription,
            facial_expression=facial_expression,
            facial_reliability=facial_reliability,
        )

    if prompt_type == "zero_shot_v4":
        return build_zero_shot_prompt_v4(
            transcription=transcription,
            facial_expression=facial_expression,
            facial_reliability=facial_reliability,
        )

    if prompt_type == "few_shot":
        return build_few_shot_prompt(
            transcription=transcription,
            facial_expression=facial_expression,
            facial_reliability=facial_reliability,
        )

    if prompt_type == "text_only":
        return build_text_only_prompt(
            transcription=transcription,
        )

    raise ValueError(
        "prompt_type must be "
        "'zero_shot', 'zero_shot_v3', "
        "'zero_shot_v4', 'few_shot', "
        "or 'text_only'."
    )


def load_dataset(output_file):
    """Load the dataset and restore previous LLM results."""

    dataframe = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    if LLM_COLUMN not in dataframe.columns:
        dataframe[LLM_COLUMN] = ""

    if output_file.exists():
        previous = pd.read_csv(
            output_file,
            keep_default_na=False,
        )

        if (
            "video_url" in previous.columns
            and LLM_COLUMN in previous.columns
        ):
            previous = (
                previous
                .drop_duplicates(
                    subset="video_url",
                    keep="last",
                )
                .set_index("video_url")
            )

            for index, row in dataframe.iterrows():
                video_url = str(
                    row["video_url"]
                ).strip()

                if video_url not in previous.index:
                    continue

                value = previous.at[
                    video_url,
                    LLM_COLUMN,
                ]

                if pd.notna(value):
                    dataframe.at[
                        index,
                        LLM_COLUMN,
                    ] = value

    return dataframe


def process_reviews(
    prompt_type="zero_shot",
    limit=None,
):
    """Classify pending reviews with the selected prompt."""

    output_file = get_output_file(
        prompt_type
    )

    dataframe = load_dataset(
        output_file
    )

    pending_indices = dataframe.index[
        dataframe[LLM_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ].tolist()

    if limit is not None:
        if limit < 1:
            raise ValueError(
                "limit must be at least 1."
            )

        pending_indices = (
            pending_indices[:limit]
        )

    if not pending_indices:
        print(
            f"All {prompt_type} reviews "
            "are already classified."
        )
        return dataframe

    print(
        f"\nPrompt type: {prompt_type}"
    )

    model = load_model()

    total = len(pending_indices)

    for position, index in enumerate(
        pending_indices,
        start=1,
    ):
        row = dataframe.loc[index]

        print(
            f"\n[{position}/{total}] "
            f"Processing review {index + 1}"
        )

        prompt = build_prompt(
            prompt_type=prompt_type,
            transcription=row["transcription"],
            facial_expression=row[
                "final_facial_expression"
            ],
            facial_reliability=row[
                "facial_reliability"
            ],
        )

        prediction = classify_sentiment(
            model,
            prompt,
        )

        dataframe.at[
            index,
            LLM_COLUMN,
        ] = prediction

        dataframe.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "LLM sentiment:",
            prediction,
        )

    print(
        "\nLLM processing finished."
    )

    print(
        "Output:",
        output_file,
    )

    return dataframe