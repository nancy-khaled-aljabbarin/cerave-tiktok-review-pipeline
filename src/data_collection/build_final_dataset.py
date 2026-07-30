from pathlib import Path

import pandas as pd


DATA_FOLDER = Path("data")

BATCH_1 = DATA_FOLDER / "review_ready_batch1.csv"
BATCH_2 = DATA_FOLDER / "review_tiktok_videos_batch2.csv"
OUTPUT_FILE = DATA_FOLDER / "cerave_reviews_final.csv"

TARGET_COUNTS = {
    "positive": 17,
    "negative": 17,
    "neutral": 16,
}


def load_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return pd.read_csv(file_path)


def main():
    batch1 = load_file(BATCH_1)
    batch2 = load_file(BATCH_2)

    data = pd.concat(
        [batch1, batch2],
        ignore_index=True,
    )

    # Keep only manually accepted reviews
    data = data[
        data["manual_keep"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    ].copy()

    # Normalize sentiment labels
    data["final_sentiment"] = (
        data["final_sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"mixed": "neutral"})
    )

    # Keep only valid sentiment classes
    data = data[
        data["final_sentiment"].isin(
            ["positive", "negative", "neutral"]
        )
    ].copy()

    # Clean only the URL
    data["video_url"] = (
        data["video_url"]
        .astype(str)
        .str.strip()
        .str.split("?")
        .str[0]
    )

    # Remove duplicate videos
    data.drop_duplicates(
        subset=["video_url"],
        keep="first",
        inplace=True,
    )

    # Keep descriptions exactly as collected
    data["video_description"] = (
        data["video_description"]
        .fillna("")
    )

    # Convert likes to numbers
    data["like_count"] = pd.to_numeric(
        data["like_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    selected_groups = []

    for sentiment, needed in TARGET_COUNTS.items():
        group = data[
            data["final_sentiment"] == sentiment
        ].copy()

        print(
            f"{sentiment}: "
            f"available={len(group)}, "
            f"needed={needed}"
        )

        if len(group) < needed:
            raise ValueError(
                f"Not enough {sentiment} videos."
            )

        # Prefer rows with descriptions, then higher likes
        group["has_description"] = (
            group["video_description"]
            .astype(str)
            .str.strip()
            .ne("")
        )

        group.sort_values(
            by=["has_description", "like_count"],
            ascending=[False, False],
            inplace=True,
        )

        selected_groups.append(
            group.head(needed)
        )

    final_data = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    # Shuffle rows with a fixed seed
    final_data = final_data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    # Keep only the final columns
    final_data = final_data[
        [
            "video_url",
            "video_description",
            "like_count",
            "final_sentiment",
        ]
    ].rename(
        columns={
            "final_sentiment": "sentiment",
        }
    )

    final_data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinal dataset created.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(final_data)}")

    print("\nSentiment counts:")
    print(final_data["sentiment"].value_counts())

    print("\nDuplicate URLs:")
    print(final_data["video_url"].duplicated().sum())

    print("\nMissing values:")
    print(final_data.isna().sum())

    print("\nEmpty descriptions:")
    print(
        final_data["video_description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


if __name__ == "__main__":
    main()