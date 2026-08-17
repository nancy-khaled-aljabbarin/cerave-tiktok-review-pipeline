def build_zero_shot_prompt(
    transcription,
    facial_expression,
    facial_reliability,
):
    return f"""
You are analyzing the overall sentiment of a TikTok review about a CeraVe skincare product.

Classify the review as exactly one of:
positive, neutral, or negative.

Sentiment definitions:
- positive: the overall opinion is clearly favorable toward the product.
- negative: the overall opinion is clearly unfavorable toward the product.
- neutral: the review is balanced, mixed, uncertain, or does not clearly lean positive or negative.

Use the transcription as the primary evidence for the sentiment decision.

Use the facial expression only as supporting evidence.
If the facial-expression reliability is low, give the facial expression little weight.
If the facial expression is unavailable or not applicable, base the decision on the transcription.

Transcription:
{transcription}

Facial expression:
{facial_expression}

Facial-expression reliability:
{facial_reliability}

Return exactly one label:
positive, neutral, or negative.
""".strip()


def build_few_shot_prompt(
    transcription,
    facial_expression,
    facial_reliability,
):
    return f"""
You are analyzing the overall sentiment of a TikTok review about a CeraVe skincare product.

Classify the review as exactly one of:
positive, neutral, or negative.

Sentiment definitions:
- positive: the overall opinion is clearly favorable toward the product.
- negative: the overall opinion is clearly unfavorable toward the product.
- neutral: the review is balanced, mixed, uncertain, or does not clearly lean positive or negative.

Use the transcription as the primary evidence for the sentiment decision.

Use the facial expression only as supporting evidence.
If the facial-expression reliability is low, give the facial expression little weight.
If the facial expression is unavailable or not applicable, base the decision on the transcription.

Examples:

Example 1:
Transcription: The product worked really well for me and I would definitely use it again.
Facial expression: happiness
Facial-expression reliability: high
Sentiment: positive

Example 2:
Transcription: The product has some good qualities, but I also experienced several issues with it. Overall, I have mixed feelings about it.
Facial expression: neutral
Facial-expression reliability: medium
Sentiment: neutral

Example 3:
Transcription: The product did not work well for me and I would not recommend it.
Facial expression: sadness
Facial-expression reliability: high
Sentiment: negative

Now classify this review:

Transcription:
{transcription}

Facial expression:
{facial_expression}

Facial-expression reliability:
{facial_reliability}

Return exactly one label:
positive, neutral, or negative.
""".strip()


def build_text_only_prompt(
    transcription,
):
    return f"""
You are analyzing the overall sentiment of a TikTok review about a CeraVe skincare product.

Classify the review as exactly one of:
positive, neutral, or negative.

Sentiment definitions:
- positive: the overall opinion is clearly favorable toward the product.
- negative: the overall opinion is clearly unfavorable toward the product.
- neutral: the review is balanced, mixed, uncertain, or does not clearly lean positive or negative.

Use only the transcription to determine the sentiment.

Transcription:
{transcription}

Return exactly one label:
positive, neutral, or negative.
""".strip()


def build_zero_shot_prompt_v3(
    transcription,
    facial_expression,
    facial_reliability,
):
    return f"""
You are analyzing the overall sentiment of a TikTok review about a CeraVe skincare product.

Classify the review as exactly one of:
positive, neutral, or negative.

Sentiment definitions:
- positive: the overall opinion is clearly favorable toward the product.
- negative: the overall opinion is clearly unfavorable toward the product.
- neutral: the review contains a balanced or mixed opinion, is uncertain, or does not clearly lean positive or negative.

Important classification rule:
If the review contains meaningful positive and negative opinions and neither clearly dominates, classify it as neutral.
Do not classify a mixed review as negative only because it contains criticism.

Use the transcription as the primary evidence for the sentiment decision.

Use the facial expression only as supporting evidence.
If the facial-expression reliability is low, give the facial expression little weight.
If the facial expression is unavailable or not applicable, base the decision on the transcription.

Transcription:
{transcription}

Facial expression:
{facial_expression}

Facial-expression reliability:
{facial_reliability}

Return exactly one label:
positive, neutral, or negative.
""".strip()


def build_zero_shot_prompt_v4(
    transcription,
    facial_expression,
    facial_reliability,
):
    return f"""
You are analyzing the overall sentiment of a TikTok review about a CeraVe skincare product.

Classify the review as exactly one of:
positive, neutral, or negative.

Sentiment definitions:
- positive: the reviewer clearly expresses an overall favorable opinion, recommendation, satisfaction, or preference toward the reviewed product.
- negative: the reviewer clearly expresses an overall unfavorable opinion, dissatisfaction, rejection, warning, or recommendation against the reviewed product.
- neutral: the review is mixed, balanced, mainly descriptive, comparative without a clear overall preference, uncertain, or does not contain a clearly dominant positive or negative opinion.

Decision rules:
1. Use the transcription as the primary evidence.
2. Determine the overall conclusion of the review, not the sentiment of individual words or sentences.
3. If both meaningful positive and negative opinions are present and neither clearly dominates, classify the review as neutral.
4. Do not classify a review as negative only because it contains criticism if the final or overall evaluation is still favorable.
5. Do not classify a review as positive only because it contains praise if the overall conclusion is unfavorable.
6. For multi-product or comparative reviews, classify according to the overall attitude of the full review. If different products receive conflicting evaluations and there is no single dominant overall sentiment, classify as neutral.
7. Mainly informational or descriptive content should be neutral unless the reviewer clearly expresses an overall positive or negative judgment.
8. Ratings, recommendations, purchase intent, willingness to reuse, and explicit final conclusions are strong sentiment evidence.
9. Implicit sentiment should only be treated as positive or negative when the overall preference is reasonably clear.

Facial-expression evidence:
- The facial expression is supporting evidence only and must not override a clear textual opinion.
- If facial-expression reliability is high, it may provide secondary support when the transcription is ambiguous.
- If facial-expression reliability is medium, use it only as weak supporting evidence.
- If facial-expression reliability is low, ignore it for the final sentiment decision.
- If facial expression is unavailable or not applicable, use only the transcription.

Transcription:
{transcription}

Facial expression:
{facial_expression}

Facial-expression reliability:
{facial_reliability}

Return exactly one label:
positive, neutral, or negative.
""".strip()