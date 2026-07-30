import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "tiny.en",
)

VERIFICATION_MODEL_NAME = os.getenv(
    "VERIFICATION_MODEL_NAME",
    "medium",
)

DEVICE = os.getenv(
    "DEVICE",
    "cpu",
)

COMPUTE_TYPE = os.getenv(
    "COMPUTE_TYPE",
    "int8",
)

TEST_LIMIT = int(
    os.getenv(
        "TEST_LIMIT",
        "50",
    )
)


DATA_FOLDER = PROJECT_ROOT / "data"

INPUT_CSV = (
    DATA_FOLDER
    / "cerave_reviews_final.csv"
)

OUTPUT_CSV = (
    DATA_FOLDER
    / "cerave_reviews_enriched.csv"
)

VIDEOS_FOLDER = (
    DATA_FOLDER
    / "videos"
)

AUDIOS_FOLDER = (
    DATA_FOLDER
    / "audios"
)


VIDEOS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIOS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)