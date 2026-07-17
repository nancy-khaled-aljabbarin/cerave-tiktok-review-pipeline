from pathlib import Path

import pandas as pd


OLD_FILE = Path("data/cerave_reviews.csv")
REVIEW_FILE = Path("data/review_tiktok_videos_batch1.csv")
OUTPUT_FILE = Path("data/review_ready_batch1.csv")


def clean_url(value):
    if pd.isna(value):
        return ""

    return str(value).split("?")[0].strip()


def main():
    old_data = pd.read_csv(
        OLD_FILE,
        keep_default_na=False,
    )

    review_data = pd.read_csv(
        REVIEW_FILE,
        keep_default_na=False,
    )

    old_data["clean_url"] = old_data[
        "video_url"
    ].apply(clean_url)

    review_data["clean_url"] = review_data[
        "video_url"
    ].apply(clean_url)

    old_labels = old_data[
        ["clean_url", "sentiment", "language"]
    ].rename(
        columns={
            "sentiment": "old_sentiment",
            "language": "old_language",
        }
    )

    review_data = review_data.merge(
        old_labels,
        on="clean_url",
        how="left",
    )

    # Keep existing review columns if they already exist.
    if "manual_keep" not in review_data.columns:
        review_data["manual_keep"] = ""

    if "final_sentiment" not in review_data.columns:
        review_data["final_sentiment"] = ""

    old_sentiment = (
        review_data["old_sentiment"]
        .str.strip()
        .str.lower()
    )

    old_language = (
        review_data["old_language"]
        .str.strip()
        .str.lower()
    )

    valid_sentiments = [
        "positive",
        "negative",
        "neutral",
    ]

    confirmed_match = (
        old_sentiment.isin(valid_sentiments)
        & old_language.eq("english")
    )

    # Reuse only labels that were already reviewed
    # and confirmed as English in the old dataset.
    review_data.loc[
        confirmed_match,
        "manual_keep",
    ] = "yes"

    review_data.loc[
        confirmed_match,
        "final_sentiment",
    ] = old_sentiment[confirmed_match]

    # Reject videos detected as possible ads.
    if "possible_ad" in review_data.columns:
        ad_rows = (
            review_data["possible_ad"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        review_data.loc[
            ad_rows,
            "manual_keep",
        ] = "no"

        review_data.loc[
            ad_rows,
            "final_sentiment",
        ] = ""

    review_data.drop(
        columns=[
            "clean_url",
            "old_sentiment",
            "old_language",
        ],
        inplace=True,
    )

    review_data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMatching finished.")
    print(
        "Automatically confirmed: "
        f"{(review_data['manual_keep'] == 'yes').sum()}"
    )
    print(
        "Automatically rejected: "
        f"{(review_data['manual_keep'] == 'no').sum()}"
    )
    print(
        "Still needs review: "
        f"{(review_data['manual_keep'] == '').sum()}"
    )
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()