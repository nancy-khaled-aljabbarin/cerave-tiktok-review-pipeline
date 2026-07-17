from pathlib import Path
import webbrowser

import pandas as pd


INPUT_FILE = Path("data/cerave_reviews_final.csv")
OUTPUT_FILE = Path("data/speech_check.csv")


def main():
    data = pd.read_csv(INPUT_FILE)
    results = []

    for index, row in data.iterrows():
        url = str(row["video_url"])

        print(f"\nVideo {index + 1}/{len(data)}")
        print(url)

        webbrowser.open(url)

        answer = input(
            "Person speaking? (y=yes, n=no, s=skip): "
        ).strip().lower()

        while answer not in ["y", "n", "s"]:
            answer = input("Enter y, n, or s: ").strip().lower()

        if answer == "y":
            status = "speech"
        elif answer == "n":
            status = "no_speech"
        else:
            status = "check_later"

        results.append(
            {
                "video_url": url,
                "sentiment": row["sentiment"],
                "speech_status": status,
            }
        )

        # يحفظ بعد كل فيديو حتى ما يضيع شغلك
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    result = pd.DataFrame(results)

    print("\nFinal speech counts:")
    print(result["speech_status"].value_counts())

    print("\nVideos without spoken review:")
    print(
        result[result["speech_status"] == "no_speech"][
            ["video_url", "sentiment"]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()