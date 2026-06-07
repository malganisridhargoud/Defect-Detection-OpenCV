# DefectVision: SVM-based Defect Detection — Full Project Overview

DefectVision is an image-level defect classification system for industrial parts. The pipeline uses OpenCV for preprocessing, handcrafted feature extraction, and an SVM classifier (scikit-learn) to decide if an image is `Good` or `Defective`.

This repository intentionally uses a minimal, explainable stack (no deep-learning model) to make the solution easy to run, interpret, and deploy on CPU-only machines.

## One-line summary

Image -> OpenCV preprocessing -> handcrafted features -> feature pipeline -> SVM classifier -> Good/Defective (+ confidence)

## Tech stack and rationale

| Purpose | Tool | Why chosen |
|---|---|---|
| Language | Python | Widely used, lots of ML/CV libraries, easy to explain in interviews |
| Web UI / Demo | Streamlit | Fast to build interactive demos without front-end code |
| Image processing | OpenCV | Efficient CPU-based image ops and morphology tools |
| Feature math | NumPy, SciPy | Fast numerical ops and statistics |
| Feature extraction helpers | scikit-image | LBP, HOG helpers (used where applicable) |
| ML | scikit-learn | SVM, pipelines, CV, grid search, calibration — ideal for small datasets |
| Persistence | joblib | Simple model + scaler serialization |
| Image IO | Pillow | Robust image format support |

Why this combination: the project prioritizes interpretability, ease of setup, explainability for interviews, and reliable CPU execution. scikit-learn pipelines give reproducible training + evaluation without heavy infrastructure.

## Full app workflow (user & developer view)

User workflow (typical):

1. Prepare dataset under `data/train/good`, `data/train/defective` (and optionally `data/test/...`).
2. (Optional) Use `src/generate_demo_data.py` to populate small demo images.
3. Start demo: `streamlit run app.py` and open `http://localhost:8501`.
4. Click `Train SVM Model` to train locally. Training runs preprocessing -> feature extraction -> pipeline fit -> evaluation.
5. Upload or drop single images into the Inspector UI and run `Run Inspection` to get a PASS/FAIL decision with confidence.

Developer workflow (how code connects):

- `src/preprocess.py`: loading, resizing, grayscale/color conversion, morphological cleanup, optional cropping.
- `src/feature_extract.py`: computes shape stats, histogram/color stats, LBP, HOG, Sobel edge stats, and other handcrafted descriptors.
- `src/train.py`: builds an sklearn `Pipeline` (scaler -> PCA? -> SVM), performs stratified CV, grid search, probability calibration, then saves `models/svm_model.pkl` and `models/scaler.pkl`.
- `src/predict.py`: loads model and scaler, runs same preprocessing + features, returns label, probability, and intermediate images/plots for UI.
- `app.py`: Streamlit UI that calls `src/train.py` and `src/predict.py` functions.

Commands to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model design and feature list

1) Preprocessing (`src/preprocess.py`)
- Resize (224x224 by default) to standardize features.
- Optional cropping to focus on the part (heuristic contours or center crop).
- Convert to grayscale for many texture/edge features; keep color channels for color histograms.
- Denoise with Gaussian blur and remove small artifacts with morphological opening/closing.

2) Feature extraction (`src/feature_extract.py`)
- Shape features: contour area, convexity defects, bounding-box aspect ratio.
- Texture: Local Binary Patterns (LBP) histograms.
- Edge features: Sobel gradient statistics and counts.
- HOG: Histogram of Oriented Gradients for gradient-orientation patterns.
- Color statistics: channel means, standard deviations, histograms.
- Intensity statistics: mean, std, skewness, kurtosis.

3) Feature pipeline & modeling (`src/train.py`)
- `sklearn.preprocessing.StandardScaler` to normalize features.
- Optional `PCA` to reduce dimensionality and noise (configurable).
- `sklearn.svm.SVC` with `probability=True` and an RBF kernel by default.
- `CalibratedClassifierCV` or `sklearn.calibration.CalibratedClassifierCV` to produce reliable probabilities if desired.
- Grid search over `C` and `gamma`, with stratified K-fold CV.

4) Prediction (`src/predict.py`)
- Run same preprocessing + feature extraction.
- Apply scaler and PCA used during training.
- Predict class and probability; output intermediate images (preprocessed, edges) for UI.

## Accuracy, evaluation, and how it's implemented

Metrics produced by `src/train.py` and saved to `results/`:

- Accuracy
- Precision / Recall / F1 (per-class and macro)
- Balanced accuracy
- ROC AUC (when probability outputs exist)
- Confusion matrix (saved to `results/confusion_matrix.png`)
- Cross-validation scores plot

Implementation notes & tips to improve accuracy:

- Use stratified K-fold CV (already used) to ensure class proportions are preserved.
- Use sample weighting or class_weight='balanced' in SVM if classes are skewed.
- Use `CalibratedClassifierCV` to improve probability estimates.
- Tune `C` and `gamma` using `GridSearchCV` with a scoring metric suited to the business need (F1 or recall for defect detection).
- Consider feature selection (recursive feature elimination) or groupwise feature importance to remove noisy descriptors.
- Add image augmentation (rotations, flips, small contrast changes) to increase effective data size.
- If available, collect and include more real defective examples; SVM benefits significantly from representative examples.

Practical evaluation steps to run locally:

```bash
# train and save model
python src/train.py --data-dir data/train --save-model models/svm_model.pkl

# run predictions over test folder and produce report
python src/predict.py --model models/svm_model.pkl --input-folder data/test --report results/test_report.json
```

## Migration note: Faster R-CNN -> SVM (what changed & why)

- This repository is purposely SVM-first. If you had a prior Faster R-CNN pipeline in another branch, migration required:
    - Remove object-detection steps that produced bounding boxes.
    - Replace per-box region extraction + CNN feature vectors with whole-image preprocessing + handcrafted features.
    - Replace deep model training code with `src/train.py` SVM pipeline.
- Why the change? whole-image SVM classification is faster, needs far less data, and is easier to explain in interview/demo settings.

Files to inspect when reviewing the migration:

- `src/train.py` — SVM pipeline & CV
- `src/predict.py` — single-image inference
- `app.py` — demo UI calls

If you want, I can open these three files and align their code to the README (e.g., ensure saved filenames `models/svm_model.pkl` and `models/scaler.pkl` are used consistently).

## Interview Q&A (expanded: why/how for each major feature)

**Q: Why preprocess images?**
A: Standardized size, noise removal, and consistent contrast reduce intra-class variation and make features comparable across images.

**Q: How does preprocessing affect model accuracy?**
A: Improves signal-to-noise ratio for features. Example: poor contrast hides scratches; CLAHE (adaptive histogram equalization) can restore visibility and improve recall for defect cases.

**Q: Why choose handcrafted features vs CNN features?**
A: Handcrafted features are interpretable and work well on small datasets. CNNs require large labeled datasets and GPUs; they can overfit on small industrial datasets.

**Q: How does each feature help detect defects?**
- Shape features detect missing pieces, unexpected holes, or deformations.
- LBP captures local texture changes (scratches, abrasions).
- HOG captures edge orientation patterns (burrs, dents) that change gradient distributions.
- Color histograms spot stains or discoloration.
- Sobel gradient statistics highlight abrupt intensity changes which are often defects.

**Q: How is the SVM configured and why?**
A: RBF kernel SVM is a robust default for non-linear separable data. `C` controls margin softness and `gamma` controls kernel reach. Grid search with CV finds a good trade-off between bias and variance.

**Q: How do you handle class imbalance?**
A: Use `class_weight='balanced'` or sample-weighting, and evaluate with recall/F1 for the defect class. Also use stratified CV.

**Q: How to tune threshold for detection sensitivity?**
A: Use probability outputs (via calibration) and choose a threshold that gives acceptable recall (catch defects) while keeping false positives manageable. Plot precision-recall to pick threshold.

**Q: How to evaluate model robustness for deployment?**
A: Use cross-validation, test on held-out production-like images, run perturbation tests (brightness/blur), and measure per-condition metrics.

**Q: How to explain a single decision to stakeholders?**
A: Show intermediate preprocessed images, top contributing features (via feature importance or a small SHAP-like analysis), class probability, and nearest training examples.

## Files and where to look (quick links)

- `app.py` — demo UI and endpoints for training/prediction
- `src/preprocess.py` — core image transforms
- `src/feature_extract.py` — all feature functions
- `src/train.py` — training loop, CV, grid search, model export
- `src/predict.py` — single-image inference and batch evaluation
- `models/` — saved `svm_model.pkl`, `scaler.pkl`
- `data/` — dataset layout for training and testing

## Limitations and next steps (practical roadmap)

- If defect localization is needed, add a two-stage approach: SVM for pass/fail, and an explainable segmenter or simple object detector for bounding boxes.
- If dataset grows substantially, consider a small CNN (transfer learning) for improved accuracy.
- Add automated monitoring (collect failed examples and retrain periodically).

## How you can ask me to continue

- I can open and align `src/train.py`, `src/predict.py`, and `app.py` to the documentation.
- I can run the training script locally and produce the evaluation report.
- I can add unit tests for the feature extraction functions.

---

If you'd like, I can now: (a) open `src/train.py` and ensure it saves the model / scaler as `models/svm_model.pkl` / `models/scaler.pkl`, (b) run a quick local training pass and save evaluation outputs, or (c) add example unit tests for `src/feature_extract.py`. Which would you prefer next?
