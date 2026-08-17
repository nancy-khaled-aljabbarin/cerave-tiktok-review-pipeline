from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

TRANSCRIPTION_FILE = (
    DATA_DIR / "transcription_verification.csv"
)

FACIAL_FILE = (
    DATA_DIR / "cerave_reviews_with_two_models.csv"
)

LLM_FILE = (
    DATA_DIR / "cerave_reviews_with_llm_zero_shot.csv"
)

REPORT_FILE = (
    DATA_DIR / "final_validation_report.txt"
)


def require_file(file_path):
    """Ensure that a required validation file exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}"
        )


def require_columns(dataframe, columns, file_name):
    """Ensure that required columns exist."""

    missing = set(columns) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{file_name} is missing columns: "
            + ", ".join(sorted(missing))
        )


def validate_transcription():
    """Summarize transcription verification results."""

    require_file(TRANSCRIPTION_FILE)

    dataframe = pd.read_csv(
        TRANSCRIPTION_FILE,
        keep_default_na=False,
    )

    require_columns(
        dataframe,
        {
            "verification_status",
            "agreement_score",
        },
        TRANSCRIPTION_FILE.name,
    )

    status = (
        dataframe["verification_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    agreement = pd.to_numeric(
        dataframe["agreement_score"],
        errors="coerce",
    )

    auto_verified = int(
        status.eq("verified_auto").sum()
    )

    manual_verified = int(
        status.eq("verified_manual").sum()
    )

    verified_total = (
        auto_verified + manual_verified
    )

    total = len(dataframe)

    return {
        "total": total,
        "verified_total": verified_total,
        "auto_verified": auto_verified,
        "manual_verified": manual_verified,
        "mean_agreement": agreement.mean(),
        "median_agreement": agreement.median(),
        "minimum_agreement": agreement.min(),
        "maximum_agreement": agreement.max(),
    }


def validate_facial_expression():
    """Validate final facial-expression predictions."""

    require_file(FACIAL_FILE)

    dataframe = pd.read_csv(
        FACIAL_FILE,
        keep_default_na=False,
    )

    require_columns(
        dataframe,
        {
            "manual_facial_expression",
            "final_facial_expression",
            "facial_reliability",
        },
        FACIAL_FILE.name,
    )

    manual = (
        dataframe["manual_facial_expression"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    predicted = (
        dataframe["final_facial_expression"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_mask = manual.ne("")

    evaluated = dataframe.loc[
        valid_mask
    ].copy()

    evaluated["correct"] = (
        manual[valid_mask]
        == predicted[valid_mask]
    )

    correct = int(
        evaluated["correct"].sum()
    )

    total = len(evaluated)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    reliability_rows = []

    for reliability, group in evaluated.groupby(
        "facial_reliability",
        dropna=False,
    ):
        group_total = len(group)
        group_correct = int(
            group["correct"].sum()
        )

        group_accuracy = (
            group_correct / group_total
            if group_total > 0
            else 0.0
        )

        reliability_rows.append(
            {
                "reliability": str(reliability),
                "videos": group_total,
                "correct": group_correct,
                "accuracy": group_accuracy,
            }
        )

    confusion_matrix = pd.crosstab(
        manual[valid_mask],
        predicted[valid_mask],
        rownames=["Manual"],
        colnames=["Predicted"],
        dropna=False,
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "reliability": reliability_rows,
        "confusion_matrix": confusion_matrix,
    }


def calculate_class_metrics(
    manual,
    predicted,
    label,
):
    """Calculate precision, recall, and F1 for one label."""

    true_positive = int(
        (
            manual.eq(label)
            & predicted.eq(label)
        ).sum()
    )

    predicted_positive = int(
        predicted.eq(label).sum()
    )

    actual_positive = int(
        manual.eq(label).sum()
    )

    precision = (
        true_positive / predicted_positive
        if predicted_positive > 0
        else 0.0
    )

    recall = (
        true_positive / actual_positive
        if actual_positive > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_positive,
    }


def validate_llm():
    """Validate zero-shot LLM sentiment predictions."""

    require_file(LLM_FILE)

    dataframe = pd.read_csv(
        LLM_FILE,
        keep_default_na=False,
    )

    require_columns(
        dataframe,
        {
            "sentiment",
            "llm_sentiment",
        },
        LLM_FILE.name,
    )

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

    correct = int(
        manual.eq(predicted).sum()
    )

    total = len(dataframe)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    labels = [
        "positive",
        "neutral",
        "negative",
    ]

    class_metrics = [
        calculate_class_metrics(
            manual,
            predicted,
            label,
        )
        for label in labels
    ]

    confusion_matrix = pd.crosstab(
        manual,
        predicted,
        rownames=["Manual"],
        colnames=["Predicted"],
        dropna=False,
    )

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix,
    }


def build_report(
    transcription,
    facial,
    llm,
):
    """Create the final validation report."""

    lines = []

    lines.append(
        "FINAL PIPELINE VALIDATION REPORT"
    )
    lines.append(
        "=" * 32
    )

    lines.append("")
    lines.append(
        "1. TRANSCRIPTION VERIFICATION"
    )
    lines.append(
        "-" * 29
    )

    lines.append(
        f"Total transcriptions: "
        f"{transcription['total']}"
    )
    lines.append(
        f"Verified transcriptions: "
        f"{transcription['verified_total']}"
    )
    lines.append(
        f"Automatically verified: "
        f"{transcription['auto_verified']}"
    )
    lines.append(
        f"Manually verified: "
        f"{transcription['manual_verified']}"
    )
    lines.append(
        "Mean agreement score: "
        f"{transcription['mean_agreement']:.2%}"
    )
    lines.append(
        "Median agreement score: "
        f"{transcription['median_agreement']:.2%}"
    )
    lines.append(
        "Minimum agreement score: "
        f"{transcription['minimum_agreement']:.2%}"
    )
    lines.append(
        "Maximum agreement score: "
        f"{transcription['maximum_agreement']:.2%}"
    )

    lines.append("")
    lines.append(
        "Note: agreement score measures "
        "agreement between transcription "
        "verification outputs; it is not "
        "reported as transcription accuracy."
    )

    lines.append("")
    lines.append(
        "2. FACIAL EXPRESSION VALIDATION"
    )
    lines.append(
        "-" * 31
    )

    lines.append(
        f"Evaluated videos: "
        f"{facial['total']}"
    )
    lines.append(
        f"Correct predictions: "
        f"{facial['correct']}"
    )
    lines.append(
        f"Incorrect predictions: "
        f"{facial['incorrect']}"
    )
    lines.append(
        "Exact-match accuracy: "
        f"{facial['accuracy']:.2%}"
    )

    lines.append("")
    lines.append(
        "Accuracy by model reliability:"
    )

    for row in facial["reliability"]:
        lines.append(
            f"  {row['reliability']}: "
            f"{row['correct']}/"
            f"{row['videos']} "
            f"({row['accuracy']:.2%})"
        )

    lines.append("")
    lines.append(
        "Facial confusion matrix:"
    )
    lines.append(
        facial[
            "confusion_matrix"
        ].to_string()
    )

    lines.append("")
    lines.append(
        "3. LLM SENTIMENT VALIDATION"
    )
    lines.append(
        "-" * 27
    )

    lines.append(
        f"Evaluated reviews: "
        f"{llm['total']}"
    )
    lines.append(
        f"Correct predictions: "
        f"{llm['correct']}"
    )
    lines.append(
        f"Incorrect predictions: "
        f"{llm['incorrect']}"
    )
    lines.append(
        "Overall accuracy: "
        f"{llm['accuracy']:.2%}"
    )

    lines.append("")
    lines.append(
        "Per-class metrics:"
    )

    for row in llm["class_metrics"]:
        lines.append(
            f"  {row['label']}: "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, "
            f"support={row['support']}"
        )

    lines.append("")
    lines.append(
        "LLM confusion matrix:"
    )
    lines.append(
        llm[
            "confusion_matrix"
        ].to_string()
    )

    lines.append("")
    lines.append(
        "VALIDATION INTERPRETATION"
    )
    lines.append(
        "-" * 25
    )

    lines.append(
        "Transcription verification confirms "
        "that all available transcriptions "
        "were checked."
    )

    lines.append(
        "Facial-expression results should be "
        "treated as a supporting signal, "
        "especially when model reliability "
        "is low."
    )

    lines.append(
        "LLM sentiment performance is evaluated "
        "against the manually assigned sentiment "
        "labels."
    )

    report = "\n".join(lines)

    return "\n".join(
        line.rstrip()
        for line in report.splitlines()
    )


def main():
    """Run the complete validation stage."""

    print(
        "\nStarting final validation...\n"
    )

    transcription = (
        validate_transcription()
    )

    facial = (
        validate_facial_expression()
    )

    llm = validate_llm()

    report = build_report(
        transcription,
        facial,
        llm,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print(
        "\nValidation report saved to:"
    )
    print(REPORT_FILE)

    print(
        "\nFinal validation completed."
    )


if __name__ == "__main__":
    main()