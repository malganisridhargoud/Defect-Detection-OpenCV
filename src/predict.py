"""
predict.py
----------
Inference module. Given a single image, runs full pipeline
and returns prediction with confidence score and visual output.
"""

import cv2
import numpy as np
import time
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from preprocess import full_pipeline, to_grayscale, resize, load_image, normalize_contrast, focus_on_object
from feature_extract import extract_all_features


def predict_single(source, clf, scaler):
    """
    Run full inference on a single image.

    Args:
        source: File path, PIL Image, or numpy array
        clf: Trained R-CNN classifier
        scaler: Fitted StandardScaler

    Returns:
        dict with keys:
            label       : "Good" or "Defective"
            label_id    : 0 or 1
            confidence  : float 0-1
            inference_ms: inference time in milliseconds
            stages      : dict of intermediate images for visualization
            annotated   : BGR numpy array with bounding boxes drawn
    """
    t0 = time.time()

    img = load_image(source)
    focused = focus_on_object(img)
    resized = resize(focused)
    gray = normalize_contrast(to_grayscale(resized))
    cleaned, stages = full_pipeline(focused, return_stages=True)

    features = extract_all_features(cleaned, gray)
    features_scaled = scaler.transform([features])
    anomaly_score = float(-clf["detector"].decision_function(features_scaled)[0])
    threshold = float(clf["threshold"])
    score_std = max(float(clf.get("score_std", 1.0)), 1e-6)
    proba_defective = 1.0 / (1.0 + math.exp(-(anomaly_score - threshold) / score_std))
    proba_good = 1.0 - proba_defective
    label_id = int(anomaly_score >= threshold)
    confidence = float(max(proba_good, proba_defective))

    inference_ms = (time.time() - t0) * 1000

    label = "Good" if label_id == 0 else "Defective"
    annotated = _draw_annotations(resized.copy(), cleaned, label, confidence)

    return {
        "label": label,
        "label_id": int(label_id),
        "confidence": confidence,
        "proba_good": float(proba_good),
        "proba_defective": float(proba_defective),
        "inference_ms": inference_ms,
        "stages": stages,
        "annotated": annotated,
        "features": features,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
    }


def _draw_annotations(img: np.ndarray, cleaned: np.ndarray,
                       label: str, confidence: float) -> np.ndarray:
    """Draw bounding boxes around detected defect regions."""
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 30]

    color_rgb = (0, 200, 80) if label == "Good" else (220, 60, 60)

    display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    if label == "Defective":
        for cnt in contours[:10]:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display, (x, y), (x+w, y+h), color_rgb, 2)

    banner_h = 36
    cv2.rectangle(display, (0, 0), (224, banner_h), color_rgb, -1)
    text = f"{label}  {confidence*100:.1f}%"
    cv2.putText(display, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return display


def make_pipeline_figure(stages: dict, figsize=(14, 5)) -> plt.Figure:
    """
    Generate a matplotlib figure showing all preprocessing stages side by side.
    Used for Streamlit display.
    """
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.patch.set_facecolor("#0f1117")

    for ax, (title, img) in zip(axes, stages.items()):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    return fig
