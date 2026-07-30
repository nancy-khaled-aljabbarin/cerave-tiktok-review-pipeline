from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/raw_tiktok_videos.csv")
OUTPUT_FILE = Path("data/review_tiktok_videos_batch2.csv")


POSITIVE_KEYWORDS = [
    "love",
    "loving",
    "amazing",
    "great",
    "best",
    "recommend",
    "recommended",
    "perfect",
    "favorite",
    "favourite",
    "effective",
    "works well",
    "helped my skin",
    "holy grail",
    "10/10",
]

NEGATIVE_KEYWORDS = [
    "hate",
    "bad",
    "worst",
    "do not recommend",
    "don't recommend",
    "broke me out",
    "breakout",
    "breakouts",
    "irritated",
    "irritation",
    "burning",
    "ruined my skin",
    "not worth",
    "disappointed",
    "miss for me",
]

NEUTRAL_KEYWORDS = [
    "honest review",
    "pros and cons",
    "comparison",
    "compare",
    "versus",
    " vs ",
    "good but",
    "works but",
    "before and after",
    "ingredients",
    "difference between",
]

AD_KEYWORDS = [
    "#ad",
    "paid partnership",
    "paid partner",
    "sponsored",
    "#sponsored",
    "ceravepartner",
    "#ceravepartner",
    "gifted",
    "affiliate",
    "discount code",
    "use my code",
    "shop now",
    "link in bio",
]


def prepare_text(value):
    if pd.isna(value):
        return ""

    return " ".join(
        str(value).lower().split()
    )


def contains_keyword(text, keywords):
    return any(
        keyword in text
        for keyword in keywords
    )


def count_keywords(text, keywords):
    return sum(
        keyword in text
        for keyword in keywords
    )


def suggest_sentiment(text):
    positive_score = count_keywords(
        text,
        POSITIVE_KEYWORDS,
    )

    negative_score = count_keywords(
        text,
        NEGATIVE_KEYWORDS,
    )

    neutral_score = count_keywords(
        text,
        NEUTRAL_KEYWORDS,
    )

    if positive_score > 0 and negative_score > 0:
        return "neutral"

    scores = {
        "positive": positive_score,
        "negative": negative_score,
        "neutral": neutral_score,
    }

    best_label = max(
        scores,
        key=scores.get,
    )

    if scores[best_label] == 0:
        return "unclear"

    return best_label


def main():
    if not INPUT_FILE.exists():
        print(
            "Raw CSV file was not found. "
            "Run collect_tiktok.py first."
        )
        return

    dataset = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    dataset.drop_duplicates(
        subset=["video_url"],
        inplace=True,
    )

    descriptions = dataset[
        "video_description"
    ].apply(prepare_text)

    dataset["possible_ad"] = descriptions.apply(
        lambda text: contains_keyword(
            text,
            AD_KEYWORDS,
        )
    )

    dataset["sentiment_suggestion"] = (
        descriptions.apply(
            suggest_sentiment
        )
    )

    dataset["manual_keep"] = ""
    dataset["final_sentiment"] = ""

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFiltering finished.")
    print(f"Total candidates: {len(dataset)}")

    print("\nPossible advertisements:")
    print(
        dataset["possible_ad"]
        .value_counts()
    )

    print("\nSentiment suggestions:")
    print(
        dataset["sentiment_suggestion"]
        .value_counts()
    )

    print(f"\nReview file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()