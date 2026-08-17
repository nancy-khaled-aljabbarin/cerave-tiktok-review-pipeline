from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILES = {
    "zero_shot_v1": (
        PROJECT_ROOT
        / "data"
        / "cerave_reviews_with_llm_v1.csv"
    ),
    "zero_shot_v2": (
        PROJECT_ROOT
        / "data"
        / "cerave_reviews_with_llm_v2.csv"
    ),
    "few_shot": (
        PROJECT_ROOT
        / "data"
        / "cerave_reviews_with_llm_few_shot.csv"
    ),
}


def evaluate_results(file_path):
    """Calculate evaluation metrics for one result file."""

    dataframe = pd.read_csv(file_path)

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

    accuracy = (
        manual == predicted
    ).mean()

    confusion_matrix = pd.crosstab(
        manual,
        predicted,
        rownames=["Manual"],
        colnames=["LLM"],
        dropna=False,
    )

    class_results = pd.DataFrame(
        {
            "sentiment": manual,
            "correct": manual == predicted,
        }
    )

    class_accuracy = (
        class_results
        .groupby("sentiment")["correct"]
        .agg(["sum", "count", "mean"])
    )

    return (
        accuracy,
        confusion_matrix,
        class_accuracy,
    )


def main():
    """Compare the available prompting approaches."""

    scores = {}

    for name, file_path in RESULT_FILES.items():
        if not file_path.exists():
            print(
                f"\nSkipping {name}: "
                "result file was not found."
            )
            continue

        (
            accuracy,
            confusion_matrix,
            class_accuracy,
        ) = evaluate_results(file_path)

        scores[name] = accuracy

        print(f"\n{name}")
        print("-" * len(name))

        print(
            f"Accuracy: {accuracy:.2%}"
        )

        print("\nConfusion matrix:")
        print(confusion_matrix)

        print("\nAccuracy by class:")
        print(class_accuracy)

    if not scores:
        print(
            "\nNo LLM result files were found."
        )
        return

    best_method = max(
        scores,
        key=scores.get,
    )

    print("\nBest prompting approach:")
    print(
        f"{best_method}: "
        f"{scores[best_method]:.2%}"
    )


if __name__ == "__main__":
    main()