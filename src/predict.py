"""
Single-image inference for the SVM defect detector.
"""

import time

import cv2
import numpy as np

from feature_extract import extract_all_features
from preprocess import focus_on_object, full_pipeline, load_image, normalize_contrast, resize, to_grayscale


def predict_single(source, clf, scaler=None):
    """
    Run inspection on one image.

    Args:
        source: file path, PIL Image, or numpy array
        clf: model bundle loaded by train.load_model
        scaler: kept for backward-compatible call sites; unused by v2 model
    """
    if clf is None or "model" not in clf:
        raise ValueError("SVM model is not loaded. Train the model first.")

    t0 = time.time()

    img = load_image(source)
    focused = focus_on_object(img)
    resized = resize(focused)
    gray = normalize_contrast(to_grayscale(resized))
    cleaned, stages = full_pipeline(focused, return_stages=True)

    features = extract_all_features(cleaned, gray, resized)
    model = clf["model"]
    probabilities = model.predict_proba([features])[0]
    proba_good = float(probabilities[0])
    proba_defective = float(probabilities[1])
    threshold = float(clf.get("defect_threshold", 0.5))
    label_id = int(proba_defective >= threshold)
    confidence = float(probabilities[label_id])
    inference_ms = (time.time() - t0) * 1000

    label = "Good" if label_id == 0 else "Defective"
    annotated = _draw_annotations(resized.copy(), cleaned, label, confidence)

    return {
        "label": label,
        "label_id": label_id,
        "confidence": confidence,
        "proba_good": proba_good,
        "proba_defective": proba_defective,
        "inference_ms": inference_ms,
        "stages": stages,
        "annotated": annotated,
        "features": features,
        "defect_threshold": threshold,
    }


def _draw_annotations(
    img: np.ndarray,
    cleaned: np.ndarray,
    label: str,
    confidence: float,
) -> np.ndarray:
    """Draw a clear pass/fail banner and candidate defect regions."""
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    color_rgb = (0, 170, 95) if label == "Good" else (220, 60, 60)
    display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    if label == "Defective":
        for cnt in contours[:12]:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x, y), (x + w, y + h), color_rgb, 2)

    banner_h = 38
    cv2.rectangle(display, (0, 0), (224, banner_h), color_rgb, -1)
    text = f"{label}  {confidence * 100:.1f}%"
    cv2.putText(
        display,
        text,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return display
