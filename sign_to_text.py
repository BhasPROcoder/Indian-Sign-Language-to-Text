"""Train and use a simple Indian Sign Language A-Z classifier.

The training images are read directly from a ZIP whose layout is:

    A/image1.jpg
    A/image2.jpg
    ...
    Z/imageN.jpg

Training extracts the coloured hand-skeleton overlay already present in the
dataset, then uses HOG features and a linear Support Vector Machine. Webcam
mode draws the same skeleton from detected hand landmarks. This prevents the
classifier from learning faces, skin tone, clothing, and room backgrounds.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import io
import json
import string
import sys
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog


IMAGE_SIZE = (64, 64)
HOG_OPTIONS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_LABELS = set(string.ascii_uppercase)
DEFAULT_IMAGES_PER_LETTER = 400
DEFAULT_MIN_SCORE = -0.25
DEFAULT_MIN_MARGIN = 0.20

FINGER_CONNECTIONS = (
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)


def fit_skeleton(mask: np.ndarray) -> np.ndarray:
    """Crop, scale, and centre a skeleton without stretching its shape."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError("No usable hand skeleton was found.")
    cropped = Image.fromarray(
        (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8)
    )
    scale = min(
        (IMAGE_SIZE[0] - 6) / cropped.width,
        (IMAGE_SIZE[1] - 6) / cropped.height,
    )
    fitted_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    cropped = cropped.resize(fitted_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", IMAGE_SIZE)
    canvas.paste(
        cropped,
        ((IMAGE_SIZE[0] - fitted_size[0]) // 2, (IMAGE_SIZE[1] - fitted_size[1]) // 2),
    )
    return np.asarray(canvas, dtype=np.uint8)


def dataset_skeleton_hog(image: Image.Image) -> np.ndarray:
    """Extract the coloured landmark overlay from one training image."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    brightness = rgb.max(axis=2)
    mask = (saturation > 35) & (brightness > 90)
    return hog(fit_skeleton(mask), **HOG_OPTIONS).astype(np.float32)


def live_skeleton_hog(hand_landmarks, frame_size: tuple[int, int]) -> np.ndarray:
    """Draw detected landmarks in the same form as the dataset overlays."""
    width, height = frame_size
    canvas = Image.new("L", (width, height))
    draw = ImageDraw.Draw(canvas)
    for hand in hand_landmarks[:2]:
        points = [(round(point.x * width), round(point.y * height)) for point in hand]
        for start, end in FINGER_CONNECTIONS:
            draw.line((points[start], points[end]), fill=255, width=2)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=255)
    return hog(fit_skeleton(np.asarray(canvas) > 0), **HOG_OPTIONS).astype(np.float32)


def label_from_zip_path(name: str) -> str | None:
    """Return A-Z from a path such as A/example.jpg; ignore other files."""
    path = Path(name)
    if path.suffix.lower() not in VALID_EXTENSIONS or not path.parts:
        return None
    label = path.parts[0].upper()
    return label if label in EXPECTED_LABELS else None


def load_features(zip_path: Path, limit_per_class: int | None = None):
    """Read the ZIP one image at a time, so it never needs to be extracted."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    counts = {label: 0 for label in sorted(EXPECTED_LABELS)}

    with zipfile.ZipFile(zip_path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and label_from_zip_path(entry.filename) is not None
        ]
        if not entries:
            raise ValueError("No images were found in A-Z folders inside the ZIP.")

        print(f"Found {len(entries):,} image files in {zip_path.name}.")
        for entry in entries:
            label = label_from_zip_path(entry.filename)
            assert label is not None
            if limit_per_class is not None and counts[label] >= limit_per_class:
                continue

            try:
                with archive.open(entry) as raw_file:
                    with Image.open(io.BytesIO(raw_file.read())) as image:
                        features.append(dataset_skeleton_hog(image))
                labels.append(label)
                counts[label] += 1
            except (OSError, UnidentifiedImageError, ValueError) as error:
                skipped.append(f"{entry.filename}: {error}")

            if len(labels) and len(labels) % 250 == 0:
                print(f"Prepared {len(labels):,} images...", end="\r", flush=True)

    print(f"Prepared {len(labels):,} readable images.        ")
    present_labels = {label for label, count in counts.items() if count}
    missing = sorted(EXPECTED_LABELS - present_labels)
    if missing:
        raise ValueError(f"The ZIP is missing these letter folders: {', '.join(missing)}")
    if min(counts.values()) < 7:
        raise ValueError("Each letter needs at least 7 readable images for the data split.")

    return np.vstack(features), np.asarray(labels), counts, skipped


def split_data(features: np.ndarray, labels: np.ndarray, random_state: int):
    """Create stratified 70% training, 15% validation and 15% test sets."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 13))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        xticks_rotation="vertical",
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Indian Sign Language alphabet confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    zip_path = args.zip.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, counts, skipped = load_features(zip_path, args.limit_per_class)
    split = split_data(features, labels, args.random_state)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split
    print(
        f"Split: {len(y_train):,} training, {len(y_validation):,} validation, "
        f"{len(y_test):,} testing"
    )

    model = LinearSVC(C=1.0, dual="auto", max_iter=10_000, random_state=args.random_state)
    print("Training the linear SVM...")
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)
    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    report = classification_report(y_test, test_predictions, digits=4, zero_division=0)

    print(f"Validation accuracy: {validation_accuracy:.2%}")
    print(f"Test accuracy:       {test_accuracy:.2%}")
    print("\nTest classification report:\n")
    print(report)

    labels_in_model = model.classes_.tolist()
    model_path = output_dir / "isl_skeleton_svm.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_type": "colored_skeleton_hog_v1",
            "image_size": IMAGE_SIZE,
            "hog_options": HOG_OPTIONS,
            "labels": labels_in_model,
        },
        model_path,
    )

    report_path = output_dir / "evaluation.txt"
    report_path.write_text(
        f"Validation accuracy: {validation_accuracy:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )
    save_confusion_matrix(
        y_test,
        test_predictions,
        labels_in_model,
        output_dir / "confusion_matrix.png",
    )

    metadata = {
        "dataset_zip": str(zip_path),
        "images_used": int(len(labels)),
        "images_per_class": counts,
        "skipped_images": skipped,
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_validation)),
        "test_images": int(len(y_test)),
        "validation_accuracy": float(validation_accuracy),
        "test_accuracy": float(test_accuracy),
        "random_state": args.random_state,
    }
    (output_dir / "run_details.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved the model and evaluation files in: {output_dir}")


def predict(args: argparse.Namespace) -> None:
    try:
        import mediapipe as mp
    except ImportError as error:
        raise SystemExit(
            "Image prediction needs MediaPipe. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    bundle = joblib.load(args.model)
    if bundle.get("feature_type") != "colored_skeleton_hog_v1":
        raise SystemExit(
            "This is an older background-sensitive model. Use "
            "results\\isl_skeleton_svm.joblib instead."
        )
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
    if not result.hand_landmarks:
        raise SystemExit("No hand was detected in the image.")
    feature = live_skeleton_hog(
        result.hand_landmarks, (rgb.shape[1], rgb.shape[0])
    ).reshape(1, -1)
    prediction = bundle["model"].predict(feature)[0]
    print(f"Predicted letter: {prediction}")


def hand_region(frame_shape: tuple[int, ...], hand_landmarks) -> tuple[int, int, int, int]:
    """Move a 4:3 classification area to the detected hand(s)."""
    height, width = frame_shape[:2]
    xs = [point.x * width for hand in hand_landmarks for point in hand]
    ys = [point.y * height for hand in hand_landmarks for point in hand]
    hand_width = max(xs) - min(xs)
    hand_height = max(ys) - min(ys)
    region_width = min(
        width,
        int(height * 4 / 3),
        max(int(width * 0.70), int(hand_width * 1.8), int(hand_height * 2.4)),
    )
    region_height = int(region_width * 3 / 4)
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2 + region_height * 0.05
    left = min(max(0, int(center_x - region_width / 2)), width - region_width)
    top = min(max(0, int(center_y - region_height / 2)), height - region_height)
    return left, top, left + region_width, top + region_height


def stable_letter(recent: deque[str]) -> str | None:
    """Show a letter only after it wins most of the recent scans."""
    if len(recent) < recent.maxlen:
        return None
    letter, votes = Counter(recent).most_common(1)[0]
    return letter if votes > len(recent) // 2 else None


def webcam(args: argparse.Namespace) -> None:
    """Predict only when a hand is detected and one letter stands out."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise SystemExit(
            "Webcam mode needs OpenCV and MediaPipe. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train first with the 'train' command."
        )
    if not args.hand_model.is_file():
        raise FileNotFoundError(
            f"Hand detector model not found: {args.hand_model}. "
            "Keep the models folder beside sign_to_text.py."
        )
    if args.scan_every < 1 or args.votes < 1:
        raise ValueError("--scan-every and --votes must both be at least 1.")

    bundle = joblib.load(args.model)
    if bundle.get("feature_type") != "colored_skeleton_hog_v1":
        raise SystemExit(
            "This is an older background-sensitive model. Use "
            "results\\isl_skeleton_svm.joblib instead."
        )
    model = bundle["model"]
    detector_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {args.camera}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    recent: deque[str] = deque(maxlen=args.votes)
    scanned_frames = 0
    timestamp_ms = 0
    sentence = ""
    status = "No hand"
    region = None
    print("Camera started. Show the whole sign anywhere in the frame.")
    print("Space: add displayed letter | Backspace: delete | C: clear | Q: quit")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(detector_options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped supplying frames.")

                if scanned_frames % args.scan_every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb),
                    )
                    timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                    hands = detector.detect_for_video(mp_image, timestamp_ms)
                    if not hands.hand_landmarks:
                        recent.clear()
                        status = "No hand"
                        region = None
                    else:
                        region = hand_region(frame.shape, hands.hand_landmarks)
                        left, top, right, bottom = region
                        # Classify only the normalized hand skeleton. The room,
                        # face, clothing, and hand position are intentionally ignored.
                        feature = live_skeleton_hog(
                            hands.hand_landmarks,
                            (frame.shape[1], frame.shape[0]),
                        ).reshape(1, -1)
                        scores = model.decision_function(feature)[0]
                        best = int(np.argmax(scores))
                        runner_up = float(np.partition(scores, -2)[-2])
                        margin = float(scores[best] - runner_up)
                        if scores[best] < args.min_score or margin < args.min_margin:
                            recent.clear()
                            status = "Uncertain"
                        else:
                            recent.append(str(model.classes_[best]))
                            status = "Scanning"
                scanned_frames += 1

                letter = stable_letter(recent)
                color = (0, 200, 0) if letter else (0, 180, 255)
                if region is not None:
                    left, top, right, bottom = region
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame, f"Letter: {letter or status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
                )
                cv2.putText(
                    frame, f"Text: {sentence[-35:]}", (20, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.putText(
                    frame, "SPACE add | BACKSPACE delete | C clear | Q quit",
                    (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1,
                )
                cv2.imshow("Indian Sign Language to Text", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and letter:
                    sentence += letter
                    print(f"Text: {sentence}")
                    recent.clear()
                elif key in (8, 127):
                    sentence = sentence[:-1]
                    print(f"Text: {sentence}")
                elif key == ord("c"):
                    sentence = ""
                    print("Text cleared.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"Final text: {sentence}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Indian Sign Language A-Z recognition using hand skeletons."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate the model.")
    train_parser.add_argument(
        "--zip", type=Path,
        default=Path("Indian_Sign_Language_Small.zip"),
        help="Dataset ZIP path (default: Indian_Sign_Language_Small.zip).",
    )
    train_parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Result folder."
    )
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_LETTER,
        help="Maximum images used per letter (default: 400).",
    )
    train_parser.set_defaults(function=train)

    predict_parser = commands.add_parser("predict", help="Predict one image.")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument(
        "--hand-model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "hand_landmarker.task",
        help="Path to the included hand detector model.",
    )
    predict_parser.set_defaults(function=predict)

    webcam_parser = commands.add_parser("webcam", help="Recognise signs from a live camera.")
    webcam_parser.add_argument("--model", type=Path, required=True)
    webcam_parser.add_argument(
        "--hand-model", type=Path,
        default=Path(__file__).resolve().parent / "models" / "hand_landmarker.task",
        help="Path to the included hand detector model.",
    )
    webcam_parser.add_argument("--camera", type=int, default=0, help="Camera number (default: 0).")
    webcam_parser.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE,
        help="Minimum letter score to display (default: -0.25).",
    )
    webcam_parser.add_argument(
        "--min-margin", type=float, default=DEFAULT_MIN_MARGIN,
        help="Minimum lead over the second-best letter (default: 0.20).",
    )
    webcam_parser.add_argument(
        "--scan-every", type=int, default=2,
        help="Predict once every N displayed frames (default: 2).",
    )
    webcam_parser.add_argument(
        "--votes", type=int, default=3,
        help="Number of recent predictions used to steady the letter (default: 3).",
    )
    webcam_parser.set_defaults(function=webcam)
    return parser


def main() -> None:
    if len(sys.argv) == 1:
        model_path = Path(__file__).resolve().parent / "results" / "isl_skeleton_svm.joblib"
        args = build_parser().parse_args(["webcam", "--model", str(model_path)])
    else:
        args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

