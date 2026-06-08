# DefectVision: SVM-based Defect Detection — Project Overview (Simplified)

DefectVision classifies whole-part images as `Good` or `Defective` using a small, explainable pipeline built on OpenCV and scikit-learn. The implementation intentionally keeps the tech stack minimal so the project is easy to run, explain, and deploy on CPU-only machines.

## One-line summary

Image -> OpenCV preprocessing -> handcrafted features (HOG, shape, color, basic stats) -> StandardScaler -> SVM -> PASS/FAIL (+ confidence)

## Reduced tech stack (what's kept)

- Python (3.10+)
- OpenCV (image preprocessing and HOG)
- NumPy (numerics)
- scikit-learn (StandardScaler, SVC, StratifiedKFold, cross_val_score)
- Streamlit (lightweight demo UI)
- Pillow (image IO)

Removed (not used anymore): PCA, GridSearchCV, CalibratedClassifierCV, LBP, Sobel-statistics, skewness/kurtosis, recursive feature elimination (RFE), SHAP-like explanations, and explicit probability calibration. The goal is clarity for a small resume project.

## Simple app workflow (plain words)

User steps:
1. Put labeled images under `data/train/good` and `data/train/defective`. Optionally add `data/test/good` and `data/test/defective` for held-out evaluation.
2. Run `streamlit run app.py` and open the UI in your browser.
3. From the sidebar, click `Train SVM Model` to train the model locally. Training will extract features from images, scale them, run cross-validation to report scores, then fit a final SVM on the full training set and save it.
4. In the Inspect tab, upload an image or choose the demo buttons to run an inspection. The app shows PASS/FAIL, a confidence score, and an annotated image.

Developer view (how components connect):
- `src/preprocess.py`: image loading, resizing, contrast normalization, and morphological cleanup.
- `src/feature_extract.py`: shape descriptors, color statistics, simple grayscale stats, and HOG features.
- `src/train.py`: builds `StandardScaler` + `SVC(probability=True)`, reports CV scores, fits on full data, and saves `models/svm_model.pkl` and a `scaler.pkl`.
- `src/predict.py`: loads the saved bundle, runs the same preprocessing + features, and returns label, probability, and annotated output for the UI.

## Model design and training details (concise)

- Features: shape features, color means/std, percentiles, entropy, and HOG descriptors.
- Scaling: `StandardScaler` applied to feature vectors.
- Classifier: `sklearn.svm.SVC` with an RBF kernel and `class_weight='balanced'` to reduce class-bias.
- Validation: stratified K-fold cross-validation (`StratifiedKFold`) and `cross_val_score` to report mean/std CV F1.
- Hyperparameters: manually choose `C` and `gamma` if needed (this repo uses sensible defaults to keep the code simple).

## Current model metrics (collected on local dataset)

- Training set: 288 samples (220 good, 68 defective)
- Train accuracy: 97.22%
- Train F1 (weighted): 0.9727
- CV F1 (mean ± std): 0.8683 ± 0.0420
- Held-out test set: 47 samples (22 good, 25 defective)
- Test accuracy: 76.60%
- Test F1 (weighted): 0.7537
- Test confusion matrix (rows=actual Good/Defective, cols=pred Good/Defective):

    [[12 10]
     [ 1 24]]

Notes: the model shows strong fit on training data but lower generalization to the small held-out test set — common with limited data. Adding more representative defective examples and simple augmentations will generally improve test performance.

## How to run locally (quick)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or to run the training script directly (developer):

```bash
python -m src.train data/train
```

## Why these simplifications

- Remove PCA/GridSearch/CalibratedClassifierCV to keep the code easy to explain during interviews and avoid introducing concepts that aren't essential for a small dataset.
- Remove LBP/Sobel/skewness/kurtosis and SHAP to focus on HOG, shape and color features which are easier to justify and reason about.

If you want any metric visualizations (precision-recall curve, confusion matrix image) embedded in the README or results folder, I can generate and add them.

---

If you'd like me to (A) add a short section to the README describing how to interpret the confusion matrix, (B) include a small example image and its annotated output, or (C) run a quick augmentation+retrain experiment, tell me which and I'll proceed.
