import re


PRODUCTS = {
    "Foaming Facial Cleanser": [
        "foaming facial cleanser",
        "foaming cleanser",
    ],
    "Hydrating Facial Cleanser": [
        "hydrating facial cleanser",
        "hydrating cleanser",
    ],
    "SA Smoothing Cleanser": [
        "sa smoothing cleanser",
        "sa cleanser",
        "salicylic acid cleanser",
    ],
    "Acne Foaming Cream Cleanser": [
        "acne foaming cream cleanser",
        "acne cleanser",
    ],
    "Moisturizing Cream": [
        "moisturizing cream",
        "moisturising cream",
        "cerave cream",
    ],
    "Daily Moisturizing Lotion": [
        "daily moisturizing lotion",
        "daily moisturising lotion",
        "moisturizing lotion",
        "moisturising lotion",
    ],
    "AM Facial Moisturizing Lotion": [
        "am facial moisturizing lotion",
        "am moisturizer",
        "am lotion",
    ],
    "PM Facial Moisturizing Lotion": [
        "pm facial moisturizing lotion",
        "pm moisturizer",
        "pm lotion",
    ],
    "Hydrating Mineral Sunscreen": [
        "hydrating mineral sunscreen",
        "mineral sunscreen",
    ],
}


SKIN_TYPES = {
    "dry": [
        "dry skin",
        "very dry",
    ],
    "oily": [
        "oily skin",
        "very oily",
    ],
    "combination": [
        "combination skin",
    ],
    "sensitive": [
        "sensitive skin",
    ],
    "normal": [
        "normal skin",
    ],
}


SKIN_CONCERNS = {
    "acne": [
        "acne",
        "pimple",
        "pimples",
        "breakout",
        "breakouts",
        "broke me out",
    ],
    "dryness": [
        "dryness",
        "dry patches",
        "dehydrated skin",
        "dehydrated",
    ],
    "irritation": [
        "irritation",
        "irritated",
        "burning",
        "stinging",
        "itchy",
    ],
    "redness": [
        "redness",
        "red skin",
    ],
    "white_cast": [
        "white cast",
    ],
    "dark_spots": [
        "dark spots",
        "hyperpigmentation",
    ],
    "damaged_skin_barrier": [
        "damaged skin barrier",
        "skin barrier",
    ],
}


INGREDIENTS = {
    "ceramides": [
        "ceramide",
        "ceramides",
    ],
    "niacinamide": [
        "niacinamide",
    ],
    "salicylic_acid": [
        "salicylic acid",
    ],
    "hyaluronic_acid": [
        "hyaluronic acid",
    ],
    "benzoyl_peroxide": [
        "benzoyl peroxide",
    ],
    "retinol": [
        "retinol",
    ],
    "vitamin_c": [
        "vitamin c",
    ],
    "spf": [
        "spf",
    ],
}


POSITIVE_RECOMMENDATION_PHRASES = [
    "i recommend",
    "highly recommend",
    "would recommend",
    "you should try",
    "worth it",
    "works for me",
    "worked for me",
    "i love",
    "love this",
    "my favorite",
    "favorite product",
    "must have",
    "holy grail",
    "repurchase",
    "buy again",
    "amazing",
    "10 out of 10",
    "10/10",
]


NEGATIVE_RECOMMENDATION_PHRASES = [
    "do not recommend",
    "don't recommend",
    "would not recommend",
    "not worth it",
    "not for me",
    "wouldn't buy",
    "never again",
    "broke me out",
    "irritated my skin",
    "hate it",
    "terrible",
    "awful",
    "trash",
    "worst",
    "waste of money",
    "returning this",
    "didn't work",
    "doesn't work",
]


def normalize_text(value: object) -> str:
    """Converts text to a clean lowercase form for matching."""
    if value is None:
        return ""

    text = str(value).lower()
    text = text.replace("’", "'")

    # Keep letters, numbers, spaces, apostrophes, slashes, and hyphens.
    text = re.sub(r"[^a-z0-9\s'/-]", " ", text)

    return " ".join(text.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Checks whether a complete word or phrase exists in the text."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def find_all_matches(
    text: str,
    categories: dict[str, list[str]],
) -> list[str]:
    """Returns every category that has at least one matching phrase."""
    matches = []

    for category, phrases in categories.items():
        if any(contains_phrase(text, phrase) for phrase in phrases):
            matches.append(category)

    return matches


def extract_recommendation(text: str) -> str:
    """Detects whether the speaker recommends the product."""
    positive_found = any(
        contains_phrase(text, phrase)
        for phrase in POSITIVE_RECOMMENDATION_PHRASES
    )

    negative_found = any(
        contains_phrase(text, phrase)
        for phrase in NEGATIVE_RECOMMENDATION_PHRASES
    )

    if positive_found and negative_found:
        return "mixed"

    if positive_found:
        return "recommended"

    if negative_found:
        return "not_recommended"

    return "unclear"


def extract_usage_duration(text: str) -> str:
    """Extracts a mentioned usage duration, such as 'for 4 years'."""
    duration_pattern = (
        r"\b(?:for\s+)?"
        r"(?:about\s+|almost\s+|nearly\s+|over\s+)?"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+"
        r"(day|days|week|weeks|month|months|year|years)"
        r"\b"
    )

    match = re.search(duration_pattern, text)

    if not match:
        return ""

    return match.group(0)


def extract_features(
    transcript: str,
    description: str,
) -> dict[str, object]:
    """Extracts structured skincare features from text."""
    normalized_transcript = normalize_text(transcript)
    normalized_description = normalize_text(description)

    combined_text = (
        f"{normalized_transcript} {normalized_description}"
    ).strip()

    products = find_all_matches(combined_text, PRODUCTS)
    skin_types = find_all_matches(combined_text, SKIN_TYPES)
    concerns = find_all_matches(combined_text, SKIN_CONCERNS)
    ingredients = find_all_matches(combined_text, INGREDIENTS)

    return {
        "product_name": (
            "; ".join(products)
            if products
            else "unknown"
        ),
        "skin_type": (
            "; ".join(skin_types)
            if skin_types
            else "unknown"
        ),
        "skin_concern": (
            "; ".join(concerns)
            if concerns
            else "unknown"
        ),
        "mentioned_ingredients": (
            "; ".join(ingredients)
            if ingredients
            else "none"
        ),
        "recommendation_intent": extract_recommendation(
            combined_text
        ),
        "usage_duration": extract_usage_duration(
            combined_text
        ),
        "transcript_word_count": len(
            normalized_transcript.split()
        ),
        "description_word_count": len(
            normalized_description.split()
        ),
    }