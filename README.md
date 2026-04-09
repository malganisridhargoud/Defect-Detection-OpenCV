# DefectVision: Industrial Defect Detection

DefectVision is a Streamlit-based inspection app for detecting surface defects in industrial part images and videos. It includes two model paths in one UI:

- A classic anomaly-detection pipeline built on hand-crafted image features and `OneClassSVM`
- A deep-learning detector built with Faster R-CNN and a MobileNet-V3-Large-FPN backbone optimized for CPU-friendly training

The project supports synthetic demo data generation, model training, single-image inspection, video analysis, pipeline visualization, and session-level inspection history.

## What This Project Does

DefectVision is designed for fast experimentation with industrial inspection workflows when you want an end-to-end app instead of a notebook. It lets you:

- Generate a complete demo dataset with good and defective samples
- Train either a classic SVM-based detector or a Faster R-CNN detector
- Upload an image and classify it as `Good` or `Defective`
- Upload a video and analyze defect probability over time
- Visualize preprocessing and inference stages
- Review training metrics, confusion matrices, and classification reports
- Track inspection history during the current Streamlit session

## Core Features

- Dual-model workflow
- Sidebar-driven model selection between `Classic (SVM)` and `R-CNN (Deep Learning)`
- Synthetic dataset generator for quick demos without external data
- Support for real datasets placed in `data/train/good` and `data/train/defective`
- Single-image inspection with annotated output, confidence, inference latency, and class probabilities
- Video inspection with frame skipping, defect timeline, per-frame table, and annotated video download
- Training dashboards for both model families
- Preprocessing and pipeline stage visualization
- Session statistics including inspected count, defect rate, and history trend chart
- Saved artifacts for trained models and result plots

## How The Two Models Work

### 1. Classic SVM Pipeline

This path is lightweight and fast, and is a good fit when you want a compact traditional computer-vision baseline.

Workflow:

1. Focus on the main object in the image to reduce background mismatch.
2. Resize to `224 x 224`.
3. Convert to grayscale.
4. Normalize local contrast using CLAHE.
5. Apply Gaussian blur.
6. Build edge and threshold maps.
7. Clean the binary mask with morphology.
8. Extract a 58-dimensional feature vector from shape, texture, edge, and grayscale statistics.
9. Train a `OneClassSVM` on good samples only.
10. Calibrate a defect threshold using labeled data and evaluate on train/test splits.

Outputs:

- `models/svm_model.pkl`
- `models/scaler.pkl`
- `results/confusion_matrix.png`
- `results/cv_scores.png`
- `results/test_confusion_matrix.png`

### 2. Faster R-CNN Pipeline

This path is for deep-learning-based defect localization and image-level defect detection.

Workflow:

1. Focus on the main object and resize to `224 x 224`.
2. Build a Faster R-CNN model with a MobileNet-V3-Large-FPN backbone.
3. Freeze the backbone and train only the detection head for faster CPU training.
4. Auto-generate bounding boxes for defective training samples by extracting contours from preprocessed defect masks.
5. Run inference and classify the image as defective if detections above the confidence threshold are present.
6. Return image-level probabilities plus defect boxes and scores.

Outputs:

- `models/rcnn_model.pth`
- `results/rcnn_loss_curve.png`
- `results/rcnn_confusion_matrix.png`

## App Overview

The app lives in [app.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/app.py) and is organized around a sidebar plus five tabs.

### Sidebar

- Choose active model: `Classic (SVM)` or `R-CNN (Deep Learning)`
- Check whether the currently selected model is already trained
- Generate demo dataset
- Train the selected model
- Review session stats and clear session history

### Inspect

- Upload a part image in `png`, `jpg`, `jpeg`, or `bmp`
- Or generate an instant demo image using `Good Part` / `Defective Part`
- Run inspection with the currently selected model
- View pass/fail result, annotated output, confidence, inference time, estimated FPS, and class probabilities

### Video Analysis

- Upload a video in `mp4`, `avi`, `mov`, or `mkv`
- Configure frame skipping to trade off speed vs temporal detail
- Analyze video using the active model
- View duration, frames analyzed, defect rate, FPS, timeline chart, and per-frame table
- Download an annotated output video

### Training Results

- SVM metrics: accuracy, F1, cross-validation, confusion matrix, reports, dataset distribution
- R-CNN metrics: test accuracy, test F1, final loss, loss curve, confusion matrix, report, dataset distribution

### Pipeline Viewer

- SVM preprocessing stages from original image to cleaned mask
- R-CNN input stages plus detection details

### Batch History

- Session-level inspection table
- Confidence trend chart across inspections

## Project Structure

```text
defect_detection/
|-- README.md
|-- app.py
|-- requirements.txt
|-- data/
|   |-- train/
|   |   |-- good/
|   |   `-- defective/
|   `-- test/
|       |-- good/
|       `-- defective/
|-- models/
|   |-- svm_model.pkl
|   |-- scaler.pkl
|   `-- rcnn_model.pth
|-- results/
|   |-- confusion_matrix.png
|   |-- cv_scores.png
|   |-- test_confusion_matrix.png
|   |-- rcnn_loss_curve.png
|   `-- rcnn_confusion_matrix.png
`-- src/
    |-- preprocess.py
    |-- feature_extract.py
    |-- train.py
    |-- predict.py
    |-- train_rcnn.py
    |-- predict_rcnn.py
    |-- video_inference.py
    |-- generate_demo_data.py
    `-- rcnn_model.py
```

## Installation

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Main packages used by the project include `streamlit`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `scikit-image`, `joblib`, `torch`, `torchvision`, and `Pillow`.

## Running The App

From this directory:

```powershell
streamlit run app.py
```

## Using Your Own Dataset

Place your real images like this:

```text
data/
|-- train/
|   |-- good/
|   `-- defective/
`-- test/
    |-- good/
    `-- defective/
```

Supported image formats:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

Notes:

- Training uses `data/train/good` and `data/train/defective`
- Test metrics use `data/test/good` and `data/test/defective` when available
- Retrain the model whenever your inspection images differ significantly from the training distribution

## Demo Dataset Generation

The synthetic generator creates metal-like surfaces with both clean and defective samples.

Generated defect styles include:

- Scratch
- Crack
- Hole
- Stain
- Dent

The UI currently generates `400` good images and `400` defective images, then splits them into train/test folders.

## Typical End-to-End Workflow

1. Install dependencies.
2. Run `streamlit run app.py`.
3. Click `Generate Demo Dataset`, or place your own dataset in the expected folders.
4. Choose `Classic (SVM)` or `R-CNN (Deep Learning)`.
5. Train the selected model.
6. Use the `Inspect` tab for single images.
7. Use the `Video Analysis` tab for videos.
8. Review outputs in `Training Results`, `Pipeline Viewer`, and `Batch History`.

## Main Source Files

- [app.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/app.py): Streamlit UI and workflow orchestration
- [src/preprocess.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/preprocess.py): Image preprocessing pipeline
- [src/feature_extract.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/feature_extract.py): Hand-crafted feature extraction for SVM
- [src/train.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/train.py): SVM training and evaluation
- [src/predict.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/predict.py): SVM single-image inference
- [src/train_rcnn.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/train_rcnn.py): Faster R-CNN training and evaluation
- [src/predict_rcnn.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/predict_rcnn.py): Faster R-CNN inference
- [src/video_inference.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/video_inference.py): Video analysis pipeline
- [src/generate_demo_data.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/generate_demo_data.py): Synthetic dataset generation

## Summary

DefectVision is a complete defect-inspection demo application with dataset generation, traditional and deep-learning model options, single-image and video inference, visual analytics, and saved artifacts in a local Streamlit workflow.
