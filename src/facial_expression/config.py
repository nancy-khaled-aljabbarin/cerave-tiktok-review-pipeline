from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
MODEL_DIR = PROJECT_ROOT / "models" / "facial_expression"

INPUT_CSV = DATA_DIR / "cerave_reviews_enriched.csv"
OUTPUT_CSV = (
    DATA_DIR
    / "cerave_reviews_with_facial_expression.csv"
)


# Hugging Face model
MODEL_REPO_ID = (
    "ElenaRyumina/face_emotion_recognition"
)

STATIC_MODEL_FILENAME = (
    "FER_static_ResNet50_AffectNet.pt"
)

DYNAMIC_MODEL_FILENAME = (
    "FER_dinamic_LSTM_Aff-Wild2.pt"
)


# Facial-expression labels
EMOTION_LABELS = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
}


# Video analysis settings
SEQUENCE_LENGTH = 10

# Analyze five representative parts of each video.
WINDOW_POSITIONS = (
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
)

# Check enough frames to find a valid
# sequence of clear face frames.
MAX_FRAMES_PER_WINDOW = 40
MIN_VALID_FRAMES = 8


# Strict automatic reliability rules
CONFIDENCE_THRESHOLD = 0.60

# At least four of the five valid windows
# must agree with the final expression.
MIN_WINDOW_AGREEMENT = 4

# The highest expression probability must
# exceed the second-highest by at least 15%.
MIN_CONFIDENCE_MARGIN = 0.15


# The current computer runs the model on CPU.
DEVICE = "cpu"