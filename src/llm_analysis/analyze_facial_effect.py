from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_ONLY_FILE = (
    PROJECT_ROOT
    / "data"
    / "cerave_reviews_with_llm_text_only.csv"
)

FACE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cerave_reviews_with_llm_v2.csv"
)


def main():
    """Analyze the effect of facial information on LLM predictions."""

    text_only = pd.read_csv(TEXT_ONLY_FILE)
    with_face = pd.read_csv(FACE_FILE)

    comparison = pd.DataFrame(
        {
            "sentiment": (
                text_only["sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
            "text_only": (
                text_only["llm_sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
            "with_face": (
                with_face["llm_sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
            "facial_reliability": (
                with_face["facial_reliability"]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
        }
    )

    comparison["text_correct"] = (
        comparison["text_only"]
        == comparison["sentiment"]
    )

    comparison["face_correct"] = (
        comparison["with_face"]
        == comparison["sentiment"]
    )

    helped = (
        ~comparison["text_correct"]
        & comparison["face_correct"]
    )

    harmed = (
        comparison["text_correct"]
        & ~comparison["face_correct"]
    )

    print(
        "Face helped:",
        helped.sum(),
    )

    print(
        "Face harmed:",
        harmed.sum(),
    )

    print(
        "No change:",
        len(comparison)
        - helped.sum()
        - harmed.sum(),
    )

    print(
        "\nAccuracy by facial reliability:"
    )

    for reliability, group in comparison.groupby(
        "facial_reliability"
    ):
        text_accuracy = (
            group["text_correct"].mean()
        )

        face_accuracy = (
            group["face_correct"].mean()
        )

        print(
            f"\n{reliability}"
        )

        print(
            f"Text only: {text_accuracy:.2%}"
        )

        print(
            f"Text + face: {face_accuracy:.2%}"
        )

    comparison["hybrid_prediction"] = (
        comparison["text_only"]
    )

    medium_mask = (
        comparison["facial_reliability"]
        == "medium"
    )

    comparison.loc[
        medium_mask,
        "hybrid_prediction",
    ] = comparison.loc[
        medium_mask,
        "with_face",
    ]

    hybrid_accuracy = (
        comparison["hybrid_prediction"]
        == comparison["sentiment"]
    ).mean()

    print(
        "\nHybrid strategy:"
    )

    print(
        "Use text + face when reliability "
        "is medium; otherwise use text only."
    )

    print(
        f"Hybrid accuracy: "
        f"{hybrid_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()