"""
Train the supervised SVM defect detector.

Expected dataset:
    data/train/good
    data/train/defective
    data/test/good          optional
    data/test/defective     optional
"""

import os

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extract import FEATURE_VECTOR_LENGTH, extract_all_features
from preprocess import focus_on_object, full_pipeline, load_image, normalize_contrast, resize, to_grayscale


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CLASS_NAMES = ["Good", "Defective"]
MODEL_VERSION = "minimal_svm_v1"
DEFECT_THRESHOLD = 0.25


def load_dataset(data_dir: str):
    """Load images from good/defective folders into feature vectors and labels."""
    X, y, paths = [], [], []

    for label_name, label_id in [("good", 0), ("defective", 1)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            print(f"Warning: folder not found: {folder}")
            continue

        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )
        print(f"Loading {len(files)} {label_name} images...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                X.append(extract_features_from_image(fpath))
                y.append(label_id)
                paths.append(fpath)
            except Exception as exc:
                print(f"  Skipping {fname}: {exc}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def extract_features_from_image(source) -> np.ndarray:
    """Run the production preprocessing path and return one SVM feature vector."""
    img = load_image(source)
    focused = focus_on_object(img)
    resized = resize(focused)
    gray = normalize_contrast(to_grayscale(resized))
    cleaned, _ = full_pipeline(focused, return_stages=True)
    return extract_all_features(cleaned, gray, resized)


def _cv_splitter(y: np.ndarray) -> StratifiedKFold:
    min_class_count = int(np.min(np.bincount(y)))
    n_splits = max(2, min(5, min_class_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def build_pipeline() -> Pipeline:
    """Create a simple scaler + SVM pipeline (no PCA, no calibration).

    We keep `probability=True` so the UI can show class probabilities, but
    avoid calibrated wrappers for simplicity in this resume-style project.
    """
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm),
    ])


def train_model(data_dir: str, progress_callback=None):
    """Train, tune, evaluate, and save the supervised SVM classifier."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if progress_callback:
        progress_callback(0.1, "Loading training images...")

    X, y, _ = load_dataset(data_dir)
    if len(X) == 0:
        raise ValueError("No images found. Expected data/train/good and data/train/defective.")
    if len(np.unique(y)) < 2:
        raise ValueError("SVM training needs both good and defective images.")
    if len(X) < 6:
        raise ValueError("Need at least 6 images for reliable SVM training.")

    if progress_callback:
        progress_callback(0.35, f"Extracted {len(X)} feature vectors. Tuning SVM...")

    cv = _cv_splitter(y)
    # Manual parameter choice: keep defaults or pass C/gamma via function args later.
    pipeline = build_pipeline()
    # Report cross-validation performance for the chosen pipeline
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    # Fit on full training data
    pipeline.fit(X, y)
    model = pipeline

    if progress_callback:
        progress_callback(0.7, "Evaluating trained model...")

    defect_threshold = DEFECT_THRESHOLD
    train_results = evaluate_features(X, y, model, threshold=defect_threshold)

    model_bundle = {
        "mode": MODEL_VERSION,
        "feature_length": FEATURE_VECTOR_LENGTH,
        "class_names": CLASS_NAMES,
        "model": model,
        "best_params": {"C": getattr(model.named_steps['svm'], 'C', None), "gamma": getattr(model.named_steps['svm'], 'gamma', None)},
        "best_cv_score": float(np.mean(cv_scores)) if len(cv_scores) > 0 else None,
        "defect_threshold": float(defect_threshold),
    }

    if progress_callback:
        progress_callback(0.85, "Saving model and reports...")

    joblib.dump(model_bundle, os.path.join(MODELS_DIR, "svm_model.pkl"))
    # Keep a small compatibility artifact for older UI checks and easy inspection.
    joblib.dump(model.named_steps["scaler"], os.path.join(MODELS_DIR, "scaler.pkl"))

    cm_path = _plot_confusion_matrix(train_results["confusion_matrix"], RESULTS_DIR)
    cv_path = _plot_cv_scores(np.array(cv_scores, dtype=np.float32), RESULTS_DIR)

    test_dir = os.path.join(os.path.dirname(data_dir), "test")
    test_results = evaluate_dataset(test_dir, model_bundle)
    test_cm_path = _plot_confusion_matrix(
        test_results["confusion_matrix"],
        RESULTS_DIR,
        filename="test_confusion_matrix.png",
        title="Held-Out Test Confusion Matrix",
    ) if test_results else None

    if progress_callback:
        progress_callback(1.0, "Training complete.")

    return {
        "n_samples": len(X),
        "n_good": int(np.sum(y == 0)),
        "n_defective": int(np.sum(y == 1)),
        "accuracy": train_results["accuracy"],
        "balanced_accuracy": train_results["balanced_accuracy"],
        "f1_score": train_results["f1_score"],
        "precision": train_results["precision"],
        "recall": train_results["recall"],
        "roc_auc": train_results["roc_auc"],
        "report": train_results["report"],
        "cm_path": cm_path,
        "cv_path": cv_path,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": [float(score) for score in cv_scores],
        "best_params": model_bundle["best_params"],
        "best_cv_score": model_bundle["best_cv_score"],
        "defect_threshold": float(defect_threshold),
        "test_results": test_results,
        "test_cm_path": test_cm_path,
    }


def evaluate_dataset(data_dir: str, model_bundle: dict):
    """Evaluate a saved model bundle on a good/defective image folder."""
    if not os.path.isdir(data_dir):
        return None

    X_eval, y_eval, _ = load_dataset(data_dir)
    if len(X_eval) == 0 or len(np.unique(y_eval)) < 2:
        return None

    return evaluate_features(
        X_eval,
        y_eval,
        model_bundle["model"],
        threshold=float(model_bundle.get("defect_threshold", 0.5)),
    )


def evaluate_features(X: np.ndarray, y: np.ndarray, model: Pipeline, threshold: float = 0.5) -> dict:
    """Calculate industry-style classification metrics."""
    probabilities = model.predict_proba(X)[:, 1]
    y_pred = (probabilities >= threshold).astype(np.int32)

    return {
        "n_samples": int(len(X)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred, average="weighted")),
        "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "roc_auc": _safe_auc(y, probabilities),
        "report": classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]),
        "defect_threshold": float(threshold),
    }


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray):
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return None


def _plot_confusion_matrix(
    cm: np.ndarray,
    save_dir: str,
    filename: str = "confusion_matrix.png",
    title: str = "Training Confusion Matrix",
) -> str:
    canvas = np.full((520, 620, 3), (17, 24, 39), dtype=np.uint8)
    white = (245, 245, 245)
    blue = (180, 120, 40)
    cell_w, cell_h = 170, 130
    start_x, start_y = 210, 145
    max_value = max(int(np.max(cm)), 1)

    cv2.putText(canvas, title, (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
    cv2.putText(canvas, "Predicted", (285, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2)
    cv2.putText(canvas, "Actual", (40, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2)

    for idx, name in enumerate(CLASS_NAMES):
        cv2.putText(canvas, name, (start_x + idx * cell_w + 25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        cv2.putText(canvas, name, (105, start_y + idx * cell_h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)

    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            intensity = int(60 + 155 * value / max_value)
            color = (intensity, 90, 45)
            x1 = start_x + col * cell_w
            y1 = start_y + row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), blue, 2)
            cv2.putText(canvas, str(value), (x1 + 72, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, white, 3)

    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, canvas)
    return path


def _plot_cv_scores(scores: np.ndarray, save_dir: str) -> str:
    canvas = np.full((440, 720, 3), (17, 24, 39), dtype=np.uint8)
    white = (245, 245, 245)
    axis = (150, 150, 150)
    bar_color = (230, 120, 37)
    mean_color = (35, 170, 245)
    left, bottom, top = 70, 360, 80
    chart_w = 610

    cv2.putText(canvas, "Cross-Validation F1 Scores", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2)
    cv2.line(canvas, (left, top), (left, bottom), axis, 2)
    cv2.line(canvas, (left, bottom), (left + chart_w, bottom), axis, 2)

    if len(scores) == 0:
        scores = np.array([0.0], dtype=np.float32)

    gap = 18
    bar_w = max(28, int((chart_w - gap * (len(scores) + 1)) / len(scores)))
    for idx, score in enumerate(scores):
        x1 = left + gap + idx * (bar_w + gap)
        y1 = int(bottom - float(score) * (bottom - top))
        cv2.rectangle(canvas, (x1, y1), (x1 + bar_w, bottom), bar_color, -1)
        cv2.putText(canvas, f"{float(score):.2f}", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
        cv2.putText(canvas, f"F{idx + 1}", (x1 + 4, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)

    mean_y = int(bottom - float(np.mean(scores)) * (bottom - top))
    cv2.line(canvas, (left, mean_y), (left + chart_w, mean_y), mean_color, 2)
    cv2.putText(canvas, f"Mean {float(np.mean(scores)):.3f}", (left + chart_w - 150, mean_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mean_color, 1)

    path = os.path.join(save_dir, "cv_scores.png")
    cv2.imwrite(path, canvas)
    return path


def load_model():
    """Load the supervised SVM bundle from disk and reject incompatible artifacts."""
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    if not os.path.exists(model_path):
        return None, None

    model_bundle = joblib.load(model_path)
    if not isinstance(model_bundle, dict):
        return None, None
    if model_bundle.get("mode") != MODEL_VERSION:
        return None, None
    if model_bundle.get("feature_length") != FEATURE_VECTOR_LENGTH:
        return None, None
    if "model" not in model_bundle:
        return None, None

    return model_bundle, None


if __name__ == "__main__":
    import sys

    train_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    results = train_model(train_dir)
    print(f"Model saved. Accuracy: {results['accuracy']:.3f}")
    print(f"Best parameters: {results['best_params']}")
