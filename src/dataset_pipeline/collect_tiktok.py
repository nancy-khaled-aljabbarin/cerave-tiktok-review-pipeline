import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from playwright.sync_api import sync_playwright


# This second batch focuses more on negative and neutral reviews
# because these classes are underrepresented in the first batch.
TARGETS = {
    "positive": 10,
    "negative": 30,
    "neutral": 30,
}

NUMBER_OF_SCROLLS = 10

# These groups only organize the search.
# They are not the final sentiment labels.
SEARCH_QUERIES = {
    "positive": [
        "CeraVe positive review",
        "CeraVe worked for me",
        "CeraVe helped my acne",
        "CeraVe before and after",
        "CeraVe favorite cleanser",
        "CeraVe moisturizer results",
    ],

    "negative": [
        "CeraVe negative review",
        "CeraVe broke me out",
        "CeraVe bad reaction",
        "CeraVe irritated my skin",
        "CeraVe ruined my skin",
        "CeraVe sunscreen white cast",
        "CeraVe cleanser not for me",
        "CeraVe product fail",
    ],

    "neutral": [
        "CeraVe honest review",
        "CeraVe pros and cons",
        "CeraVe comparison",
        "CeraVe vs Cetaphil",
        "CeraVe cleanser review",
        "CeraVe sunscreen review",
        "CeraVe moisturizer review",
        "CeraVe skincare routine review",
    ],
}

DATA_FOLDER = Path("data")
OUTPUT_FILE = DATA_FOLDER / "raw_tiktok_videos.csv"


def clean_url(url):
    if not url:
        return ""

    return url.split("?")[0].strip()


def clean_text(text):
    if not text:
        return ""

    return " ".join(text.split())


def convert_like_count(value):
    if not value:
        return ""

    value = value.strip().upper().replace(",", "")

    try:
        if value.endswith("K"):
            return int(float(value[:-1]) * 1_000)

        if value.endswith("M"):
            return int(float(value[:-1]) * 1_000_000)

        if value.endswith("B"):
            return int(float(value[:-1]) * 1_000_000_000)

        return int(float(value))

    except ValueError:
        return ""


def collect_links(page):
    collected_links = []
    seen_links = set()

    for category, queries in SEARCH_QUERIES.items():
        category_count = 0
        category_target = TARGETS[category]

        print(f"\nStarting category: {category}")

        for query in queries:
            if category_count >= category_target:
                break

            search_url = (
                "https://www.tiktok.com/search?q="
                + quote_plus(query)
            )

            print(f"\nSearching for: {query}")

            try:
                page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )

                page.wait_for_timeout(5_000)

            except Exception as error:
                print(
                    "Could not open this search: "
                    f"{type(error).__name__}"
                )
                continue

            for scroll_number in range(NUMBER_OF_SCROLLS):
                video_elements = page.locator(
                    "a[href*='/video/']"
                )

                for index in range(video_elements.count()):
                    url = video_elements.nth(
                        index
                    ).get_attribute("href")

                    url = clean_url(url)

                    if url and url not in seen_links:
                        seen_links.add(url)
                        collected_links.append(url)
                        category_count += 1

                    if category_count >= category_target:
                        break

                print(
                    f"{category} - Scroll {scroll_number + 1}: "
                    f"{category_count}/{category_target} links"
                )

                if category_count >= category_target:
                    break

                page.mouse.wheel(0, 3_000)
                page.wait_for_timeout(2_000)

        print(
            f"Finished {category}: "
            f"{category_count} collected links"
        )

    return collected_links


def get_description(page):
    meta_selectors = [
        "meta[property='og:description']",
        "meta[name='description']",
        "meta[name='twitter:description']",
    ]

    for selector in meta_selectors:
        element = page.locator(selector).first

        if element.count() > 0:
            description = clean_text(
                element.get_attribute("content")
            )

            if description:
                return description

    text_selectors = [
        "[data-e2e='browse-video-desc']",
        "[data-e2e='video-desc']",
    ]

    for selector in text_selectors:
        element = page.locator(selector).first

        if element.count() > 0:
            description = clean_text(
                element.text_content()
            )

            if description:
                return description

    return ""


def get_like_count(page):
    like_selectors = [
        "[data-e2e='like-count']",
        "[data-e2e='browse-like-count']",
        "strong[data-e2e='like-count']",
        "button[data-e2e='like-icon'] strong",
        "button[aria-label*='like' i] strong",
        "button[aria-label*='like' i] span",
    ]

    for selector in like_selectors:
        element = page.locator(selector).first

        if element.count() > 0:
            like_count = clean_text(
                element.text_content()
            )

            if like_count:
                return convert_like_count(like_count)

    like_button = page.locator(
        "button[aria-label*='like' i]"
    ).first

    if like_button.count() > 0:
        label = clean_text(
            like_button.get_attribute("aria-label")
        )

        if label:
            match = re.search(
                r"[\d,.]+[KMB]?",
                label.upper(),
            )

            if match:
                return convert_like_count(
                    match.group()
                )

    page_html = page.content()

    patterns = [
        r'"diggCount"\s*:\s*(\d+)',
        r'\\"diggCount\\"\s*:\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, page_html)

        if match:
            return int(match.group(1))

    return ""


def collect_video_info(page, video_url):
    video_data = {
        "video_url": video_url,
        "video_description": "",
        "like_count": "",
        "sentiment": "",
    }

    try:
        page.goto(
            video_url,
            wait_until="domcontentloaded",
            timeout=60_000,
        )

        page.wait_for_timeout(5_000)

        video_data["video_description"] = (
            get_description(page)
        )

        video_data["like_count"] = (
            get_like_count(page)
        )

    except Exception as error:
        print(
            "Could not read this video: "
            f"{type(error).__name__}"
        )

    return video_data


def main():
    DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            locale="en-US",
            viewport={
                "width": 1400,
                "height": 900,
            },
        )

        page = context.new_page()

        page.goto(
            "https://www.tiktok.com/",
            wait_until="domcontentloaded",
            timeout=60_000,
        )

        print("\nTikTok is open.")
        print(
            "Complete any normal login or verification "
            "if TikTok requests it."
        )

        input(
            "\nPress Enter in the terminal "
            "when TikTok is ready..."
        )

        if page.is_closed():
            print(
                "The browser page was closed. "
                "Run the script again and keep it open."
            )
            return

        video_links = collect_links(page)

        print(
            f"\nReading {len(video_links)} videos..."
        )

        videos = []

        for index, video_url in enumerate(
            video_links,
            start=1,
        ):
            print(
                f"[{index}/{len(video_links)}] "
                "Reading video"
            )

            video_data = collect_video_info(
                page,
                video_url,
            )

            videos.append(video_data)

        context.close()
        browser.close()

    dataset = pd.DataFrame(videos)

    if dataset.empty:
        print("\nNo videos were collected.")
        return

    dataset.drop_duplicates(
        subset=["video_url"],
        inplace=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCollection finished.")
    print(f"Collected videos: {len(dataset)}")

    print(
        "Empty descriptions: "
        f"{(dataset['video_description'] == '').sum()}"
    )

    print(
        "Empty like counts: "
        f"{(dataset['like_count'] == '').sum()}"
    )

    print(f"CSV file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()