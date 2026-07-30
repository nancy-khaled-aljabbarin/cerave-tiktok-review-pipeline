import re


CERAVE_MISTAKES = [
    r"\bsarah\s+v\.?\b",
    r"\bsarah\s+made\b",
    r"\bsera\s+v\.?\b",
    r"\bceravi\b",
    r"\bceravee\b",
    r"\bservey\b",
    r"\bcrb\b",
]


PRODUCT_WORDS = (
    "moisturizer",
    "cleanser",
    "lotion",
    "cream",
    "product",
    "sunscreen",
    "anti-itch",
)


SPELLING_FIXES = [
    (r"\bspy\b", "SPF"),
    (r"\bskin care\b", "skincare"),
    (r"\bhydronic acid\b", "hyaluronic acid"),
    (r"\bhydropic acid\b", "hyaluronic acid"),
    (r"\balready acid\b", "hyaluronic acid"),
    (r"\bceremonies\b", "ceramides"),
    (r"\bceramics\b", "ceramides"),
    (r"\banti-inch\b", "anti-itch"),
    (r"\bkeratosis loris\b", "keratosis pilaris"),
    (r"\bkeratosis ploris\b", "keratosis pilaris"),
    (r"\bshop lips\b", "chapped lips"),
    (r"\bsurvey\b", "CeraVe"),
]


def fix_cerave_name(text: str) -> str:
    """
    Correct common Whisper mistakes for the CeraVe brand.
    """

    if not text:
        return ""

    corrected_text = text

    for mistake_pattern in CERAVE_MISTAKES:
        corrected_text = re.sub(
            mistake_pattern,
            "CeraVe",
            corrected_text,
            flags=re.IGNORECASE,
        )

    product_pattern = "|".join(
        re.escape(word)
        for word in PRODUCT_WORDS
    )

    corrected_text = re.sub(
        rf"\b(?:serve|survey)(?=\s+(?:{product_pattern})\b)",
        "CeraVe",
        corrected_text,
        flags=re.IGNORECASE,
    )

    corrected_text = re.sub(
        r"\bserve\s+it(?=\s+(?:moisturizer|cleanser|lotion|cream))",
        "CeraVe",
        corrected_text,
        flags=re.IGNORECASE,
    )

    return corrected_text


def fix_spelling_mistakes(text: str) -> str:
    """
    Correct known skincare transcription mistakes.
    """

    if not text:
        return ""

    corrected_text = text

    for mistake_pattern, correct_word in SPELLING_FIXES:
        corrected_text = re.sub(
            mistake_pattern,
            correct_word,
            corrected_text,
            flags=re.IGNORECASE,
        )

    return corrected_text


def preprocess_text(text: str) -> str:
    """
    Apply all transcription corrections.
    """

    text = fix_cerave_name(text)
    text = fix_spelling_mistakes(text)

    return text