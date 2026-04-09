"""
video_inference.py
------------------
Frame-level defect detection on video files.
"""

import cv2
import numpy as np
import os
import tempfile
import math
from typing import Callable, Optional

from preprocess import full_pipeline, to_grayscale, resize, normalize_contrast, focus_on_object
from feature_extract import extract_all_features


def process_video(
    input_path: str,
    clf,
    scaler,
    frame_skip: int = 2,
    progress_callback: Optional[Callable] = None,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    out_fd, output_path = tempfile.mkstemp(suffix="_annotated.mp4")
    os.close(out_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_results = []
    frame_idx = 0
    last_result = {"label": "Good", "label_id": 0, "confidence": 1.0, "proba_defective": 0.0, "threshold": 0.5}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            try:
                result = _predict_frame(frame, clf, scaler)
                last_result = result
            except Exception:
                result = last_result
        else:
            result = last_result

        annotated = _annotate_frame(frame.copy(), result)
        out.write(annotated)

        if frame_idx % frame_skip == 0:
            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / fps,
                "label": result["label"],
                "label_id": result["label_id"],
                "confidence": result["confidence"],
                "proba_defective": result["proba_defective"],
            })

        frame_idx += 1

        if progress_callback and frame_idx % max(1, total_frames // 40) == 0:
            progress_callback(min(frame_idx / total_frames, 0.98), f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    if progress_callback:
        progress_callback(1.0, "Done!")

    defective_count = sum(1 for r in frame_results if r["label_id"] == 1)
    defect_rate = defective_count / max(len(frame_results), 1)

    return {
        "output_path": output_path,
        "frame_results": frame_results,
        "total_frames": total_frames,
        "processed_frames": len(frame_results),
        "fps": fps,
        "duration_s": duration_s,
        "defect_rate": defect_rate,
    }


def _predict_frame(frame: np.ndarray, clf, scaler) -> dict:
    focused = focus_on_object(frame)
    resized = resize(focused)
    gray = normalize_contrast(to_grayscale(resized))
    cleaned = full_pipeline(focused)
    features = extract_all_features(cleaned, gray)
    features_scaled = scaler.transform([features])

    anomaly_score = float(-clf["detector"].decision_function(features_scaled)[0])
    threshold = float(clf["threshold"])
    score_std = max(float(clf.get("score_std", 1.0)), 1e-6)
    proba_defective = 1.0 / (1.0 + math.exp(-(anomaly_score - threshold) / score_std))
    proba_good = 1.0 - proba_defective
    label_id = int(anomaly_score >= threshold)
    confidence = float(max(proba_good, proba_defective))

    return {
        "label": "Good" if label_id == 0 else "Defective",
        "label_id": label_id,
        "confidence": confidence,
        "proba_good": float(proba_good),
        "proba_defective": float(proba_defective),
        "threshold": threshold,
    }


def _annotate_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    is_defective = result["label_id"] == 1
    border_color = (0, 60, 220) if is_defective else (0, 200, 80)
    box_color = (0, 60, 220) if is_defective else (0, 200, 80)

    border = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border * 2)

    if is_defective:
        cleaned = full_pipeline(focus_on_object(frame))
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 30]

        scale_x = w / 224.0
        scale_y = h / 224.0

        for cnt in contours[:12]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + bw) * scale_x)
            y2 = int((y + bh) * scale_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    banner_h = max(36, h // 18)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), border_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    label_text = f"{'DEFECT' if is_defective else 'GOOD'}  {result['confidence']*100:.1f}%"
    font_scale = max(0.45, w / 800)
    cv2.putText(frame, label_text, (border + 4, banner_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def read_video_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
