from collections import Counter
from pathlib import Path

import cv2
from deepface import DeepFace

from .config import VIDEOS_DIR, WINDOW_POSITIONS


# # Map DeepFace labels to the first model labels
LABEL_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


# Twenty locations distributed across the entire video
RETRY_POSITIONS = tuple(
    index / 21
    for index in range(1, 21)
)


def create_empty_result(status):
    """Return a consistent result when no face can be analyzed."""

    return {
        "deepface_expression": "no_visible_face",
        "deepface_confidence": 0.0,
        "deepface_status": status,
        "deepface_frames_analyzed": 0,
        "deepface_agreement": "0/0",
        "deepface_model": "DeepFace",
    }


def select_largest_face(results):
    """
    When more than one face appears,
    select the largest visible face.
    """

    if isinstance(results, dict):
        return results

    if not results:
        raise ValueError("No face detected")

    return max(
        results,
        key=lambda result: (
            result.get("region", {}).get("w", 0)
            * result.get("region", {}).get("h", 0)
        ),
    )


def analyze_frame(frame, detectors):
    """
    Analyze one frame.

    The detectors are tried in order,
    and the first successful result is used.
    """

    for detector in detectors:
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=detector,
                enforce_detection=True,
                align=True,
                silent=True,
            )

            result = select_largest_face(results)

            dominant_emotion = LABEL_MAP[
                result["dominant_emotion"]
            ]

            emotion_scores = {
                LABEL_MAP[label]: float(score) / 100.0
                for label, score in result["emotion"].items()
            }

            return (
                dominant_emotion,
                emotion_scores,
                detector,
            )

        except ValueError:
            # This detector didn't find a face, let's try the next detector.
            continue

        except Exception as error:
            print(
                f"Detector {detector} failed: {error}"
            )
            continue

    raise ValueError("No face detected")


def analyze_positions(
    capture,
    total_frames,
    positions,
    detectors,
):
    """Analyze a group of positions from the video."""

    frame_labels = []
    frame_scores = []

    for frame_number, position in enumerate(
        positions,
        start=1,
    ):
        target_frame = int(
            (total_frames - 1) * position
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame,
        )

        success, frame = capture.read()

        if not success or frame is None:
            print(
                f"Frame {frame_number}: could not read"
            )
            continue

        try:
            expression, scores, detector = analyze_frame(
                frame,
                detectors,
            )

            frame_labels.append(expression)
            frame_scores.append(scores)

            print(
                f"Frame {frame_number}: "
                f"{expression} "
                f"({scores[expression]:.2%}) "
                f"using {detector}"
            )

        except ValueError:
            print(
                f"Frame {frame_number}: no face detected"
            )

    return frame_labels, frame_scores


def analyze_video_deepface(video_path):
    """
    First analyze five representative frames.

    If no face is detected, retry using twenty frames
    and multiple face detectors.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        return create_empty_result(
            status="video_not_found"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return create_empty_result(
            status="video_open_failed"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        capture.release()

        return create_empty_result(
            status="empty_video"
        )

    try:
        print("\nFirst attempt: 5 frames using RetinaFace")

        frame_labels, frame_scores = analyze_positions(
            capture=capture,
            total_frames=total_frames,
            positions=WINDOW_POSITIONS,
            detectors=("retinaface",),
        )

        # Try again only if no face appears.
        if not frame_labels:
            print(
                "\nNo face detected in the first attempt."
            )
            print(
                "Retrying with 20 frames and multiple detectors..."
            )

            frame_labels, frame_scores = analyze_positions(
                capture=capture,
                total_frames=total_frames,
                positions=RETRY_POSITIONS,
                detectors=(
                    "retinaface",
                    "opencv",
                    "mtcnn",
                ),
            )

    finally:
        capture.release()

    if not frame_labels:
        return create_empty_result(
            status="no_face"
        )

    # Calculating the average score for each expression across frames
    average_scores = {
        expression: sum(
            scores.get(expression, 0.0)
            for scores in frame_scores
        )
        / len(frame_scores)
        for expression in LABEL_MAP.values()
    }

    # # Majority vote across frames
    label_counts = Counter(frame_labels)

    highest_vote_count = max(
        label_counts.values()
    )

    tied_labels = [
        label
        for label, count in label_counts.items()
        if count == highest_vote_count
    ]

    final_expression = max(
        tied_labels,
        key=lambda label: average_scores[label],
    )

    final_confidence = average_scores[
        final_expression
    ]

    valid_frames = len(frame_labels)

    agreement_count = label_counts[
        final_expression
    ]

    return {
        "deepface_expression": final_expression,
        "deepface_confidence": round(
            final_confidence,
            4,
        ),
        "deepface_status": "success",
        "deepface_frames_analyzed": valid_frames,
        "deepface_agreement": (
            f"{agreement_count}/{valid_frames}"
        ),
        "deepface_model": "DeepFace",
    }


def main():
    """Test the improved analysis on video_024 only."""

    test_video = VIDEOS_DIR / "video_024.mp4"

    print("Analyzing:", test_video.name)

    result = analyze_video_deepface(
        test_video
    )

    print("\nDeepFace final result")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()