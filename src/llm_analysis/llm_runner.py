from pathlib import Path

from llama_cpp import Llama


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "llm"
    / "qwen2.5-3b-instruct-q4_k_m.gguf"
)

VALID_SENTIMENTS = {
    "positive",
    "neutral",
    "negative",
}


def load_model():
    """Load the local GGUF model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LLM model was not found: {MODEL_PATH}"
        )

    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        verbose=False,
    )


def extract_sentiment(text):
    """Extract one valid sentiment label from the model output."""

    normalized = (
        text
        .strip()
        .lower()
    )

    first_line = (
        normalized
        .splitlines()[0]
        .strip()
    )

    if first_line in VALID_SENTIMENTS:
        return first_line

    for sentiment in VALID_SENTIMENTS:
        if normalized.startswith(sentiment):
            return sentiment

    return "invalid"


def classify_sentiment(model, prompt):
    """Run one sentiment-classification prompt."""

    response = model.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=10,
    )

    content = (
        response["choices"][0]["message"]["content"]
    )

    return extract_sentiment(content)