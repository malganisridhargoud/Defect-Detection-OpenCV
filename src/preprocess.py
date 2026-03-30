"""
preprocess.py
-------------
Image preprocessing pipeline for industrial defect detection.
Handles grayscale conversion, resizing, noise removal, and thresholding.
"""

import cv2
import numpy as np
from PIL import Image


IMG_SIZE = (224, 224)


def load_image(source) -> np.ndarray:
    """Load image from file path or PIL Image or numpy array."""
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return img
    elif isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    elif isinstance(source, np.ndarray):
        return source
    else:
        raise TypeError(f"Unsupported image type: {type(source)}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Reduce lighting sensitivity so uploads from different cameras stay comparable."""
    gray_u8 = gray.astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)


def resize(img: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    """Resize image to fixed dimensions."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def focus_on_object(img: np.ndarray, pad_ratio: float = 0.12) -> np.ndarray:
    """
    Crop around the main foreground object to reduce background mismatch
    between training images and uploaded inspection photos.
    """
    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try both bright-on-dark and dark-on-bright assumptions.
    _, mask_light = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, mask_dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    def best_crop(mask: np.ndarray):
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 0.03 * img.shape[0] * img.shape[1]:
            return None
        return cv2.boundingRect(cnt)

    candidates = [box for box in (best_crop(mask_light), best_crop(mask_dark)) if box]
    if not candidates:
        return img

    x, y, w, h = max(candidates, key=lambda box: box[2] * box[3])
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    cropped = img[y1:y2, x1:x2]
    return cropped if cropped.size else img


def gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to suppress noise."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding handles uneven lighting on curved/reflective surfaces.
    Better than global threshold for industrial parts.
    """
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )


def sobel_edge_map(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude map."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude


def morphological_clean(img: np.ndarray) -> np.ndarray:
    """
    Close small gaps in defect contours and remove noise blobs
    using morphological closing then opening.
    """
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def full_pipeline(source, return_stages: bool = False):
    """
    Run complete preprocessing pipeline.

    Args:
        source: Image path, PIL Image, or numpy array
        return_stages: If True, return dict of all intermediate stages

    Returns:
        Preprocessed grayscale image ready for feature extraction.
        If return_stages=True, returns (final_img, stages_dict)
    """
    original = load_image(source)
    focused = focus_on_object(original)
    resized = resize(focused)
    gray = to_grayscale(resized)
    normalized = normalize_contrast(gray)
    blurred = gaussian_blur(normalized)
    thresh = adaptive_threshold(blurred)
    edges = sobel_edge_map(blurred)
    cleaned = morphological_clean(thresh)

    if return_stages:
        stages = {
            "Original": cv2.cvtColor(resize(original), cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else resize(original),
            "Object Focus": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3 else resized,
            "Grayscale": gray,
            "Contrast Normalize": normalized,
            "Gaussian Blur": blurred,
            "Sobel Edges": edges,
            "Adaptive Threshold": thresh,
            "Morphological Clean": cleaned,
        }
        return cleaned, stages

    return cleaned
