from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from PIL import Image
from torchvision import transforms

from .config import (
    CONFIDENCE_THRESHOLD,
    EMOTION_LABELS,
    MAX_FRAMES_PER_WINDOW,
    MIN_CONFIDENCE_MARGIN,
    MIN_VALID_FRAMES,
    MIN_WINDOW_AGREEMENT,
    MODEL_REPO_ID,
    SEQUENCE_LENGTH,
    VIDEOS_DIR,
    WINDOW_POSITIONS,
)
from .model_loader import load_models


def preprocess_face(face_rgb):
    """Prepare one detected face for the static model."""

    image = Image.fromarray(face_rgb)

    # Preserve facial details while resizing.
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    face_tensor = transforms.PILToTensor()(image)
    face_tensor = face_tensor.to(torch.float32)

    # Apply the preprocessing used by the original model.
    face_tensor = torch.flip(
        face_tensor,
        dims=(0,),
    )

    face_tensor[0] -= 91.4953
    face_tensor[1] -= 103.8827
    face_tensor[2] -= 131.0912

    return face_tensor.unsqueeze(0)


def get_face_box(
    face_landmarks,
    frame_width,
    frame_height,
):
    """Create a padded face bounding box inside the frame."""

    x_values = []
    y_values = []

    for landmark in face_landmarks.landmark:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        x = min(max(x, 0), frame_width - 1)
        y = min(max(y, 0), frame_height - 1)

        x_values.append(x)
        y_values.append(y)

    start_x = min(x_values)
    end_x = max(x_values)
    start_y = min(y_values)
    end_y = max(y_values)

    face_width = end_x - start_x
    face_height = end_y - start_y

    # Include the forehead, cheeks, and jaw.
    padding_x = int(face_width * 0.20)
    padding_y = int(face_height * 0.20)

    start_x = max(start_x - padding_x, 0)
    end_x = min(end_x + padding_x, frame_width)

    start_y = max(start_y - padding_y, 0)
    end_y = min(end_y + padding_y, frame_height)

    return start_x, start_y, end_x, end_y


def collect_face_features(
    capture,
    face_mesh,
    static_model,
    device,
    start_frame,
):
    """Collect facial features from one video window."""

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    features = []
    checked_frames = 0

    while (
        len(features) < SEQUENCE_LENGTH
        and checked_frames < MAX_FRAMES_PER_WINDOW
    ):
        success, frame = capture.read()

        if not success or frame is None:
            break

        checked_frames += 1

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        detection_result = face_mesh.process(
            frame_rgb
        )

        if not detection_result.multi_face_landmarks:
            continue

        face_landmarks = (
            detection_result.multi_face_landmarks[0]
        )

        frame_height, frame_width = frame_rgb.shape[:2]

        start_x, start_y, end_x, end_y = get_face_box(
            face_landmarks=face_landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        face = frame_rgb[
            start_y:end_y,
            start_x:end_x,
        ]

        if face.size == 0:
            continue

        face_tensor = preprocess_face(face).to(device)

        with torch.inference_mode():
            feature = torch.relu(
                static_model.extract_features(
                    face_tensor
                )
            )

        features.append(
            feature.squeeze(0)
        )

    return features, checked_frames


def create_empty_result(status, expression=""):
    """Create a consistent result when analysis cannot finish."""

    return {
        "facial_expression": expression,
        "facial_expression_confidence": 0.0,
        "facial_expression_status": status,
        "face_frames_analyzed": 0,
        "valid_windows": 0,
        "window_agreement": "0/0",
        "confidence_margin": 0.0,
        "facial_expression_model": MODEL_REPO_ID,
    }


def select_final_expression(
    window_labels,
    average_probabilities,
):
    """
    Select the final expression using majority voting.

    When two expressions receive the same number of votes,
    the one with the higher average probability is selected.
    """

    label_counts = Counter(window_labels)

    final_index = max(
        label_counts,
        key=lambda label: (
            label_counts[label],
            float(
                average_probabilities[label].item()
            ),
        ),
    )

    agreement_count = label_counts[final_index]

    return final_index, agreement_count


def calculate_confidence_margin(
    average_probabilities,
    final_index,
):
    """
    Calculate the difference between the selected expression
    and its strongest alternative.
    """

    final_confidence = float(
        average_probabilities[
            final_index
        ].item()
    )

    alternative_confidences = [
        float(probability.item())
        for index, probability in enumerate(
            average_probabilities
        )
        if index != final_index
    ]

    strongest_alternative = max(
        alternative_confidences,
        default=0.0,
    )

    return final_confidence - strongest_alternative


def analyze_video(
    video_path,
    static_model=None,
    dynamic_model=None,
    device=None,
):
    """Analyze the dominant visible facial expression."""

    video_path = Path(video_path)

    if not video_path.exists():
        return create_empty_result(
            status="video_not_found"
        )

    if (
        static_model is None
        or dynamic_model is None
        or device is None
    ):
        static_model, dynamic_model, device = (
            load_models()
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

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

    window_predictions = []
    window_labels = []
    total_face_frames = 0

    face_mesh_module = mp.solutions.face_mesh

    try:
        with face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            for window_number, position in enumerate(
                WINDOW_POSITIONS,
                start=1,
            ):
                start_frame = int(
                    (total_frames - 1) * position
                )

                features, checked_frames = (
                    collect_face_features(
                        capture=capture,
                        face_mesh=face_mesh,
                        static_model=static_model,
                        device=device,
                        start_frame=start_frame,
                    )
                )

                print(
                    f"Window {window_number}: "
                    f"{len(features)} face frames "
                    f"from {checked_frames} checked frames"
                )

                if len(features) < MIN_VALID_FRAMES:
                    print(
                        "Skipped: not enough face frames"
                    )
                    continue

                total_face_frames += len(features)

                # Complete a shorter valid sequence by
                # repeating its final facial feature.
                while len(features) < SEQUENCE_LENGTH:
                    features.append(
                        features[-1].clone()
                    )

                sequence = torch.stack(
                    features[:SEQUENCE_LENGTH]
                ).unsqueeze(0)

                with torch.inference_mode():
                    probabilities = dynamic_model(
                        sequence
                    ).squeeze(0)

                predicted_index = int(
                    torch.argmax(
                        probabilities
                    ).item()
                )

                confidence = float(
                    probabilities[
                        predicted_index
                    ].item()
                )

                expression = EMOTION_LABELS[
                    predicted_index
                ]

                print(
                    f"Prediction: {expression} "
                    f"({confidence:.2%})"
                )

                window_labels.append(
                    predicted_index
                )

                window_predictions.append(
                    probabilities
                )

    finally:
        capture.release()

    if not window_predictions:
        return create_empty_result(
            status="no_face",
            expression="no_face",
        )

    # Average probabilities are used for confidence
    # calculations, not as the only final decision.
    average_probabilities = torch.stack(
        window_predictions
    ).mean(dim=0)

    final_index, agreement_count = (
        select_final_expression(
            window_labels=window_labels,
            average_probabilities=average_probabilities,
        )
    )

    final_expression = EMOTION_LABELS[
        final_index
    ]

    final_confidence = float(
        average_probabilities[
            final_index
        ].item()
    )

    confidence_margin = (
        calculate_confidence_margin(
            average_probabilities=average_probabilities,
            final_index=final_index,
        )
    )

    valid_windows = len(
        window_predictions
    )

    # Accept a result only when all reliability
    # conditions are satisfied.
    is_reliable = (
        final_confidence >= CONFIDENCE_THRESHOLD
        and agreement_count >= MIN_WINDOW_AGREEMENT
        and confidence_margin >= MIN_CONFIDENCE_MARGIN
    )

    if is_reliable:
        status = "success"
    else:
        status = "uncertain"

    return {
        "facial_expression": final_expression,
        "facial_expression_confidence": round(
            final_confidence,
            4,
        ),
        "facial_expression_status": status,
        "face_frames_analyzed": total_face_frames,
        "valid_windows": valid_windows,
        "window_agreement": (
            f"{agreement_count}/{valid_windows}"
        ),
        "confidence_margin": round(
            confidence_margin,
            4,
        ),
        "facial_expression_model": MODEL_REPO_ID,
    }


def main():
    """Run a local facial-expression analysis test."""

    test_video = VIDEOS_DIR / "test_video.mp4"

    print("Loading facial-expression models...")

    static_model, dynamic_model, device = load_models()

    print("Analyzing:", test_video.name)

    result = analyze_video(
        video_path=test_video,
        static_model=static_model,
        dynamic_model=dynamic_model,
        device=device,
    )

    print("\nFinal result")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()