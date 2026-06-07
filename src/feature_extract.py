"""
Feature extraction for the SVM defect detector.

Only OpenCV and NumPy are used here to keep the project stack small.
"""

import cv2
import numpy as np


LBP_BINS = 26
HOG_LENGTH = 1764
FEATURE_VECTOR_LENGTH = 14 + LBP_BINS + 6 + 12 + 18 + HOG_LENGTH


def extract_shape_features(cleaned_img: np.ndarray) -> np.ndarray:
    """Extract contour-derived features from the binary defect mask."""
    contours, _ = cv2.findContours(
        cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 20]

    if not contours:
        return np.zeros(14, dtype=np.float32)

    areas, perimeters, aspect_ratios, solidities, extents = [], [], [], [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))

        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(float(w) / (h + 1e-6))
        solidities.append(area / (hull_area + 1e-6))
        extents.append(area / (w * h + 1e-6))

    largest = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return np.array([
        np.sum(areas),
        np.max(areas),
        np.mean(areas),
        np.sum(areas) / cleaned_img.size,
        float(len(contours)),
        np.mean(perimeters),
        np.mean(aspect_ratios),
        np.mean(solidities),
        np.mean(extents),
        *hu[:5],
    ], dtype=np.float32)


def extract_lbp_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract a compact Local Binary Pattern histogram.

    OpenCV/NumPy implementation keeps the project dependency-light.
    """
    gray = gray_img.astype(np.uint8, copy=False)
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, :-2], gray[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, 256), density=True)
    return hist.astype(np.float32)


def extract_edge_features(gray_img: np.ndarray, cleaned_img: np.ndarray) -> np.ndarray:
    """Capture gradient strength and defect-mask occupancy."""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    edge_pixels = magnitude > float(np.mean(magnitude) + np.std(magnitude))
    defect_pixels = cleaned_img > 0

    return np.array([
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.max(magnitude)),
        float(np.mean(edge_pixels)),
        float(np.mean(defect_pixels)),
        float(np.sum(edge_pixels & defect_pixels) / cleaned_img.size),
    ], dtype=np.float32)


def extract_statistical_features(gray_img: np.ndarray) -> np.ndarray:
    """Summarize grayscale distribution for lighting and texture changes."""
    pixels = gray_img.astype(np.float32).ravel()
    mean = np.mean(pixels)
    std = np.std(pixels)
    centered = (pixels - mean) / (std + 1e-6)
    percentiles = np.percentile(pixels, [5, 10, 25, 50, 75, 90, 95])
    entropy_hist, _ = np.histogram(pixels, bins=32, range=(0, 256), density=True)
    entropy = -np.sum(entropy_hist * np.log2(entropy_hist + 1e-9))

    return np.array([
        mean,
        std,
        float(np.mean(centered ** 3)),
        float(np.mean(centered ** 4)),
        *percentiles,
        float(entropy),
    ], dtype=np.float32)


def extract_color_features(bgr_img: np.ndarray) -> np.ndarray:
    """Capture color changes that can indicate stains, burns, or coating issues."""
    if len(bgr_img.shape) == 2:
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    features = []

    for image in (bgr_img, hsv, lab):
        pixels = image.reshape(-1, 3).astype(np.float32)
        features.extend(np.mean(pixels, axis=0))
        features.extend(np.std(pixels, axis=0))

    return np.array(features, dtype=np.float32)


def extract_hog_features(gray_img: np.ndarray) -> np.ndarray:
    """Extract HOG structure features using OpenCV."""
    hog = cv2.HOGDescriptor(
        _winSize=(224, 224),
        _blockSize=(56, 56),
        _blockStride=(28, 28),
        _cellSize=(28, 28),
        _nbins=9,
    )
    return hog.compute(gray_img.astype(np.uint8)).reshape(-1).astype(np.float32)


def extract_all_features(
    cleaned_img: np.ndarray,
    gray_img: np.ndarray,
    bgr_img: np.ndarray | None = None,
) -> np.ndarray:
    """Combine all features into one fixed-length SVM input vector."""
    if bgr_img is None:
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    features = np.concatenate([
        extract_shape_features(cleaned_img),
        extract_lbp_features(gray_img),
        extract_edge_features(gray_img, cleaned_img),
        extract_statistical_features(gray_img),
        extract_color_features(bgr_img),
        extract_hog_features(gray_img),
    ])

    if len(features) != FEATURE_VECTOR_LENGTH:
        raise ValueError(
            f"Feature length mismatch: expected {FEATURE_VECTOR_LENGTH}, got {len(features)}"
        )

    return np.nan_to_num(features.astype(np.float32, copy=False))
