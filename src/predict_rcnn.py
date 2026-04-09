"""
predict_rcnn.py
---------------
Inference module for the Faster R-CNN defect detection model.
Returns results in the same dict format as the existing predict.py
so the Streamlit UI can use either model interchangeably.
"""

import cv2
import numpy as np
import time
import torch
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from preprocess import load_image, focus_on_object, resize
from rcnn_model import get_device

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5


def predict_single_rcnn(source, model) -> dict:
    """
    Run R-CNN inference on a single image.

    Args:
        source: File path, PIL Image, or numpy array
        model: Trained Faster R-CNN model (in eval mode)

    Returns:
        dict matching the format of predict.predict_single():
            label, label_id, confidence, proba_good, proba_defective,
            inference_ms, stages, annotated, detections
    """
    t0 = time.time()
    device = get_device()

    # --- Preprocessing ---
    img_bgr = load_image(source)
    focused = focus_on_object(img_bgr)
    resized = resize(focused, (IMG_SIZE, IMG_SIZE))

    # Build stages dict for pipeline viewer
    stages = _build_stages(img_bgr, focused, resized)

    # Convert to tensor
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels_pred = pred["labels"].cpu().numpy()

    # Filter by confidence threshold
    high_conf_mask = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[high_conf_mask]
    scores = scores[high_conf_mask]
    labels_pred = labels_pred[high_conf_mask]

    # Determine image-level classification
    if len(scores) > 0:
        label = "Defective"
        label_id = 1
        max_score = float(scores.max())
        proba_defective = float(np.mean(scores))  # average confidence of detections
        proba_good = 1.0 - proba_defective
        confidence = max_score
    else:
        label = "Good"
        label_id = 0
        proba_defective = 0.0
        proba_good = 1.0
        confidence = 1.0

    inference_ms = (time.time() - t0) * 1000

    # Draw annotations
    annotated = _draw_rcnn_annotations(
        resized.copy(), boxes, scores, label, confidence
    )

    return {
        "label": label,
        "label_id": label_id,
        "confidence": confidence,
        "proba_good": float(proba_good),
        "proba_defective": float(proba_defective),
        "inference_ms": inference_ms,
        "stages": stages,
        "annotated": annotated,
        "detections": {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "count": len(boxes),
        },
    }


def predict_frame_rcnn(frame: np.ndarray, model) -> dict:
    """
    Lightweight R-CNN inference for video frames.

    Args:
        frame: BGR numpy array (video frame)
        model: Trained Faster R-CNN model

    Returns:
        dict with label, label_id, confidence, proba_good, proba_defective, boxes
    """
    device = get_device()

    focused = focus_on_object(frame)
    resized = resize(focused, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    scores = pred["scores"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()

    high_conf_mask = scores >= CONFIDENCE_THRESHOLD
    scores = scores[high_conf_mask]
    boxes = boxes[high_conf_mask]

    if len(scores) > 0:
        label = "Defective"
        label_id = 1
        proba_defective = float(np.mean(scores))
        proba_good = 1.0 - proba_defective
        confidence = float(scores.max())
    else:
        label = "Good"
        label_id = 0
        proba_defective = 0.0
        proba_good = 1.0
        confidence = 1.0

    return {
        "label": label,
        "label_id": label_id,
        "confidence": confidence,
        "proba_good": float(proba_good),
        "proba_defective": float(proba_defective),
        "boxes": boxes.tolist(),
    }


def _draw_rcnn_annotations(img: np.ndarray, boxes: np.ndarray,
                            scores: np.ndarray, label: str,
                            confidence: float) -> np.ndarray:
    """Draw R-CNN bounding boxes on the image."""
    display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    if label == "Defective" and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            score = scores[i] if i < len(scores) else confidence

            # Color intensity based on confidence
            intensity = int(60 + score * 160)
            color = (intensity, 40, 40)  # reddish

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Score label on box
            score_text = f"{score*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, score_text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Top banner
    color_rgb = (0, 200, 80) if label == "Good" else (220, 60, 60)
    banner_h = 36
    cv2.rectangle(display, (0, 0), (IMG_SIZE, banner_h), color_rgb, -1)
    n_det = len(boxes) if label == "Defective" else 0
    text = f"{label}  {confidence*100:.1f}%"
    if n_det > 0:
        text += f"  ({n_det} defect{'s' if n_det > 1 else ''})"
    cv2.putText(display, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return display


def _build_stages(original: np.ndarray, focused: np.ndarray,
                  resized: np.ndarray) -> dict:
    """Build the stages dict for pipeline visualization."""
    stages = {
        "Original": cv2.cvtColor(resize(original, (IMG_SIZE, IMG_SIZE)),
                                  cv2.COLOR_BGR2RGB) if len(original.shape) == 3
                   else resize(original, (IMG_SIZE, IMG_SIZE)),
        "Object Focus": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3
                       else resized,
        "R-CNN Input": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3
                      else resized,
    }
    return stages


def make_rcnn_pipeline_figure(stages: dict, detections: dict,
                               figsize=(10, 4)) -> plt.Figure:
    """
    Generate a matplotlib figure showing R-CNN pipeline stages
    and detection results.
    """
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.patch.set_facecolor("#0f1117")

    if n == 1:
        axes = [axes]

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
