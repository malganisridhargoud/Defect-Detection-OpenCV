

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_BINS = LBP_N_POINTS + 2
FEATURE_VECTOR_LENGTH = 58


def extract_shape_features(cleaned_img: np.ndarray) -> np.ndarray:
    """
    Extract contour-derived features from the binary defect mask.

    Returns:
        Feature vector of length 14.
    """
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

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(float(w) / (h + 1e-6))
        solidities.append(area / (hull_area + 1e-6))
        extents.append(area / (w * h + 1e-6))

    largest = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    features = np.array([
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
    return features


def extract_lbp_features(gray_img: np.ndarray) -> np.ndarray:
    """Extract normalized LBP histogram from contrast-normalized grayscale input."""
    lbp = local_binary_pattern(
        gray_img, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD
    )
    hist, _ = np.histogram(
        lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True
    )
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


def extract_all_features(cleaned_img: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    """
    Combine contour, texture, gradient, and grayscale statistics.

    Total length: 14 + 26 + 6 + 12 = 58
    """
    shape_feats = extract_shape_features(cleaned_img)
    lbp_feats = extract_lbp_features(gray_img)
    edge_feats = extract_edge_features(gray_img, cleaned_img)
    stat_feats = extract_statistical_features(gray_img)
    features = np.concatenate([shape_feats, lbp_feats, edge_feats, stat_feats])
    return features.astype(np.float32, copy=False)
