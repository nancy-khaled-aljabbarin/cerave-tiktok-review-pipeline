import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


DATA_DIR = PROJECT_ROOT / "data"

RAW_FILE = DATA_DIR / "raw_tiktok_videos.csv"

BATCH_1_FILE = (
    DATA_DIR
    / "review_ready_batch1.csv"
)

BATCH_2_FILE = (
    DATA_DIR
    / "review_tiktok_videos_batch2.csv"
)

FINAL_FILE = (
    DATA_DIR
    / "cerave_reviews_final.csv"
)

TRANSCRIPTION_FILE = (
    DATA_DIR
    / "cerave_reviews_enriched.csv"
)

CONDA_EXE = Path(
    os.getenv(
        "CONDA_EXE",
        Path.home()
        / "anaconda3"
        / "Scripts"
        / "conda.exe",
    )
)

FACIAL_ENV_NAME = os.getenv(
    "FACIAL_ENV_NAME",
    "facial-expression",
)

EXPECTED_SENTIMENT_COUNTS = {
    "positive": 17,
    "negative": 17,
    "neutral": 16,
}