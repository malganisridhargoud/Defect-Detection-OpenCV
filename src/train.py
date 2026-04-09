"""
train.py
--------
Train an anomaly detector using good samples as the reference distribution.
This is more robust than a closed-set classifier when uploaded defects differ
from the examples seen during training.
"""

import os
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import full_pipeline, to_grayscale, resize, load_image, normalize_contrast, focus_on_object
from feature_extract import extract_all_features, FEATURE_VECTOR_LENGTH


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_dataset(data_dir: str):
    """Load images from data_dir/good and data_dir/defective into feature vectors."""
    X, y, paths = [], [], []

    for label_name, label_id in [("good", 0), ("defective", 1)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.exists(folder):
            print(f"Warning: folder not found: {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        print(f"Loading {len(files)} {label_name} images...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                img = load_image(fpath)
                focused = focus_on_object(img)
                resized = resize(focused)
                gray = normalize_contrast(to_grayscale(resized))
                cleaned, _ = full_pipeline(focused, return_stages=True)
                X.append(extract_all_features(cleaned, gray))
                y.append(label_id)
                paths.append(fpath)
            except Exception as exc:
                print(f"  Skipping {fname}: {exc}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def build_detector():
    """Factory for the anomaly model used throughout training and inference."""
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.12)


def anomaly_scores(model, X_scaled: np.ndarray) -> np.ndarray:
    """Higher score means more anomalous / more likely defective."""
    return -model.decision_function(X_scaled).reshape(-1)


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Pick the threshold that maximizes weighted F1 on labeled calibration data.
    """
    if len(scores) == 0:
        return 0.0, 0.0

    candidate_thresholds = np.unique(np.percentile(scores, np.linspace(5, 95, 91)))
    best_threshold = float(np.median(scores))
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        preds = (scores >= threshold).astype(np.int32)
        curr_f1 = f1_score(labels, preds, average="weighted")
        if curr_f1 > best_f1:
            best_f1 = float(curr_f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def conservative_threshold(scores: np.ndarray, labels: np.ndarray, tuned_threshold: float) -> float:
    """
    Favor fewer false defect alarms by ensuring most good samples remain good.
    """
    good_scores = scores[labels == 0]
    if len(good_scores) == 0:
        return tuned_threshold
    good_guardrail = float(np.percentile(good_scores, 97))
    return max(float(tuned_threshold), good_guardrail)


def evaluate_dataset(data_dir: str, model_bundle, scaler):
    """Evaluate a trained anomaly detector on a separate dataset folder."""
    if not os.path.exists(data_dir):
        return None

    X_eval, y_eval, _ = load_dataset(data_dir)
    if len(X_eval) == 0:
        return None

    X_eval_scaled = scaler.transform(X_eval)
    scores = anomaly_scores(model_bundle["detector"], X_eval_scaled)
    y_pred = (scores >= model_bundle["threshold"]).astype(np.int32)

    return {
        "n_samples": len(X_eval),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "f1_score": float(f1_score(y_eval, y_pred, average="weighted")),
        "report": classification_report(y_eval, y_pred, target_names=["Good", "Defective"]),
        "confusion_matrix": confusion_matrix(y_eval, y_pred),
    }


def cross_validate_anomaly_model(X: np.ndarray, y: np.ndarray) -> list[float]:
    """
    Cross-validate by fitting only on good training samples and validating on both classes.
    """
    scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        good_mask = y_train == 0
        if not np.any(good_mask):
            continue

        scaler = StandardScaler()
        X_train_good_scaled = scaler.fit_transform(X_train[good_mask])
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        detector = build_detector()
        detector.fit(X_train_good_scaled)

        train_scores = anomaly_scores(detector, X_train_scaled)
        threshold, _ = calibrate_threshold(train_scores, y_train)

        val_scores = anomaly_scores(detector, X_val_scaled)
        val_pred = (val_scores >= threshold).astype(np.int32)
        scores.append(float(f1_score(y_val, val_pred, average="weighted")))

    return scores


def train_model(data_dir: str, progress_callback=None):
    """
    Train the anomaly detector and calibrate the defect threshold.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if progress_callback:
        progress_callback(0.1, "Loading dataset...")

    X, y, _ = load_dataset(data_dir)
    if len(X) == 0:
        raise ValueError("No images found. Check data directory structure.")

    good_mask = y == 0
    if not np.any(good_mask):
        raise ValueError("Need at least some good samples to learn the normal surface pattern.")

    if progress_callback:
        progress_callback(0.3, f"Loaded {len(X)} images. Normalizing features...")

    scaler = StandardScaler()
    X_good_scaled = scaler.fit_transform(X[good_mask])
    X_all_scaled = scaler.transform(X)

    if progress_callback:
        progress_callback(0.5, "Training anomaly detector on good parts...")

    detector = build_detector()
    detector.fit(X_good_scaled)

    raw_scores = anomaly_scores(detector, X_all_scaled)
    tuned_threshold, tuned_f1 = calibrate_threshold(raw_scores, y)
    threshold = conservative_threshold(raw_scores, y, tuned_threshold)
    y_pred = (raw_scores >= threshold).astype(np.int32)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    report = classification_report(y, y_pred, target_names=["Good", "Defective"])
    cm = confusion_matrix(y, y_pred)

    score_std = float(np.std(raw_scores) + 1e-6)
    model_bundle = {
        "mode": "anomaly_detector",
        "feature_length": FEATURE_VECTOR_LENGTH,
        "detector": detector,
        "threshold": float(threshold),
        "score_mean": float(np.mean(raw_scores)),
        "score_std": score_std,
        "nu": 0.12,
    }

    if progress_callback:
        progress_callback(0.7, "Running cross-validation and evaluation...")

    cv_scores = cross_validate_anomaly_model(X, y)
    cv_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
    cv_std = float(np.std(cv_scores)) if cv_scores else 0.0

    if progress_callback:
        progress_callback(0.85, "Saving model and plots...")

    joblib.dump(model_bundle, os.path.join(MODELS_DIR, "rcnn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    cm_path = _plot_confusion_matrix(cm, RESULTS_DIR)
    cv_path = _plot_cv_scores(np.array(cv_scores, dtype=np.float32), RESULTS_DIR)

    test_dir = os.path.join(os.path.dirname(data_dir), "test")
    test_results = evaluate_dataset(test_dir, model_bundle, scaler)
    test_cm_path = _plot_confusion_matrix(
        test_results["confusion_matrix"],
        RESULTS_DIR,
        filename="test_confusion_matrix.png",
        title="Test Confusion Matrix",
    ) if test_results else None

    if progress_callback:
        progress_callback(1.0, "Training complete!")

    return {
        "n_samples": len(X),
        "n_good": int(np.sum(y == 0)),
        "n_defective": int(np.sum(y == 1)),
        "accuracy": float(acc),
        "f1_score": float(f1),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_scores": cv_scores,
        "report": report,
        "cm_path": cm_path,
        "cv_path": cv_path,
        "test_results": test_results,
        "test_cm_path": test_cm_path,
        "threshold": float(threshold),
        "tuned_f1": float(tuned_f1),
    }


def _plot_confusion_matrix(
    cm: np.ndarray, save_dir: str, filename: str = "confusion_matrix.png",
    title: str = "Confusion Matrix"
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good", "Defective"],
        yticklabels=["Good", "Defective"],
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 16, "color": "white"},
    )
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def _plot_cv_scores(scores: np.ndarray, save_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    if len(scores) == 0:
        scores = np.array([0.0], dtype=np.float32)
        folds = ["Fold 1"]
    else:
        folds = [f"Fold {i+1}" for i in range(len(scores))]

    bars = ax.bar(folds, scores, color=["#3b82f6"] * len(scores), width=0.5, edgecolor="#1e3a5f")
    ax.axhline(float(np.mean(scores)), color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Mean: {float(np.mean(scores)):.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score", color="white")
    ax.set_title("5-Fold Cross Validation Scores", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e293b", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{float(score):.3f}", ha="center", color="white", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "cv_scores.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def load_model():
    """Load detector and scaler from disk, rejecting stale incompatible files."""
    model_path = os.path.join(MODELS_DIR, "rcnn_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model_bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if not isinstance(model_bundle, dict):
        return None, None
    if model_bundle.get("feature_length") != FEATURE_VECTOR_LENGTH:
        return None, None
    if model_bundle.get("mode") != "anomaly_detector":
        return None, None
    if getattr(scaler, "n_features_in_", None) not in (None, FEATURE_VECTOR_LENGTH):
        return None, None

    return model_bundle, scaler


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    results = train_model(data_dir)
    print(f"\nModel saved. Accuracy: {results['accuracy']:.3f}")
