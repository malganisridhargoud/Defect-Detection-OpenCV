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

## Abstract

Industrial quality inspection is a core requirement in manufacturing environments where defective parts can reduce reliability, increase warranty cost, and interrupt downstream assembly. Manual inspection is often slow, inconsistent, and difficult to scale when defect patterns are subtle or production throughput is high. This project presents `DefectVision`, a computer-vision-based defect inspection system that combines a classical machine-learning pipeline with a deep-learning detection pipeline inside a single interactive Streamlit application.

The system is designed to support both rapid experimentation and practical demonstration. Two complementary model families are implemented. The first is a classical anomaly-detection approach based on handcrafted visual features extracted after structured preprocessing and classified using `OneClassSVM`. This path is lightweight, fast, and suitable when good samples are easier to obtain than a complete catalog of defect classes. The second is a Faster R-CNN-based detector with a MobileNet-V3-Large-FPN backbone, intended for image-level defect detection with region localization. The deep-learning path introduces bounding-box reasoning while keeping CPU-oriented training feasible by freezing the backbone and training primarily the detection head.

The application supports synthetic dataset generation, model training, single-image inference, frame-sampled video analysis, preprocessing-stage visualization, and result reporting. The preprocessing stack includes object-centric cropping, grayscale conversion, contrast normalization, Gaussian smoothing, adaptive thresholding, Sobel edge analysis, and morphological cleaning. For the classical pipeline, these processed representations are converted into a 58-dimensional feature vector that captures shape, texture, edges, and grayscale statistics. For the R-CNN pipeline, contour-derived pseudo-boxes are generated automatically from defective samples to create detector supervision without manual annotation.

The project demonstrates how classical vision methods and deep learning can coexist in one inspection workflow. The classical model provides a strong baseline with low computational overhead, while the R-CNN path extends the system toward localization and richer detection behavior. Overall, `DefectVision` serves as an end-to-end defect detection framework for educational, prototyping, and small-scale industrial inspection scenarios.

## Literature Review

Automated defect detection has been studied extensively across industrial computer vision, especially in inspection tasks involving metal surfaces, semiconductors, textiles, and manufactured components. Traditional methods and modern deep-learning methods each address different parts of the problem, and this project reflects that split directly in its architecture.

### Classical Computer Vision For Inspection

Before the rise of deep learning, industrial defect detection relied heavily on carefully designed preprocessing and feature engineering. The logic behind these systems was straightforward: if defects alter surface texture, intensity distribution, contour continuity, or local geometry, then image transformations can expose those differences and handcrafted features can quantify them.

Common preprocessing operations in the literature include:

- Grayscale conversion to reduce color variability when structure matters more than hue
- Histogram equalization or CLAHE for handling uneven illumination
- Gaussian smoothing for suppressing acquisition noise
- Edge extraction using operators such as Sobel or Canny
- Thresholding for separating suspect regions from background texture
- Morphological opening and closing for removing small artifacts and joining broken boundaries

These operations are still useful in industrial settings because they are interpretable, fast, and often robust when the imaging setup is controlled. In this project, the SVM pipeline follows that tradition closely through object-focused cropping, contrast normalization, thresholding, edge extraction, and morphology.

### Texture And Shape-Based Feature Engineering

Handcrafted descriptors remain important in defect inspection research, especially when datasets are small. Texture descriptors such as Local Binary Pattern (`LBP`) are widely used because many surface anomalies present as local texture disruptions rather than large semantic objects. Statistical intensity features also remain valuable for capturing lighting shifts, stain-like defects, and local entropy changes. Shape features based on contours, area, perimeter, solidity, extent, and Hu moments are especially relevant when defects create cracks, holes, or fragmented regions.

The feature engineering strategy in `DefectVision` aligns with this literature. The implemented 58-dimensional representation combines:

- Contour and shape descriptors from cleaned binary masks
- LBP-based texture histograms from normalized grayscale images
- Edge statistics from Sobel magnitude maps
- Distributional grayscale statistics such as percentiles and entropy

This hybrid design reflects a common insight from classical inspection systems: no single handcrafted feature is sufficient across all defect types, but a combined descriptor can provide a useful decision surface.

### SVMs And Anomaly Detection In Quality Control

Support Vector Machines have a long history in industrial inspection because they work well on moderate-sized, structured feature vectors. In particular, `OneClassSVM` is suited for anomaly detection when normal samples are plentiful and defect samples are scarce, incomplete, or open-ended. This is a realistic assumption in production environments: manufacturers often know what a good part looks like, but defective parts may vary widely.

This project uses `OneClassSVM` not as a closed-set classifier but as a normality model. Good samples define the reference manifold, and the system learns how far a new sample deviates from that reference. This makes the classical branch more appropriate for anomaly-style inspection than a standard binary classifier trained only on a narrow defect catalog.

### Deep Learning And Region-Based Detection

Deep learning has changed defect inspection by allowing models to learn discriminative features directly from images. Convolutional neural networks outperform traditional systems in many unconstrained settings, especially when defect appearance is complex or when localization matters. Among object detectors, Faster R-CNN remains one of the most influential two-stage architectures because it balances proposal-based reasoning with strong accuracy.

For industrial inspection, object detection offers two practical advantages:

- It can identify where a defect is located instead of only predicting whether a part is defective.
- It provides region-level confidence signals that are useful for operator review and explanation.

However, deep-learning systems often demand annotated data and larger compute budgets. This project addresses both issues pragmatically. It uses a lighter Faster R-CNN backbone suitable for CPU-oriented experimentation and derives defect boxes automatically from contour analysis on defective training images. While these pseudo-labels are not equivalent to manually curated annotations, they create a practical bridge between classical preprocessing and deep-learning supervision.

### Hybrid Approaches In Modern Inspection

Recent inspection literature increasingly combines classical and deep methods instead of treating them as mutually exclusive. Classical preprocessing is still valuable for reducing noise, normalizing lighting, or generating weak supervision, while deep models absorb more complex variation at inference time. `DefectVision` follows exactly this hybrid philosophy:

- The SVM branch uses classical CV end to end.
- The R-CNN branch uses classical preprocessing to generate defect supervision.
- The Streamlit UI exposes both methods under one workflow for comparison.

This dual-path design is useful both educationally and practically. It shows the strengths of interpretable feature engineering while also demonstrating how detection-based deep learning can be layered on top when localization and richer decision behavior are needed.

## Methodology

The implementation is organized as a complete defect inspection workflow rather than as isolated training scripts. The methodology can be described across six stages: data preparation, preprocessing, feature extraction, classical model training, deep-learning model training, and deployment through the application interface.

### 1. Data Preparation

The project supports two data sources:

- A synthetic demo dataset generated by `src/generate_demo_data.py`
- A user-supplied dataset placed in `data/train/good`, `data/train/defective`, `data/test/good`, and `data/test/defective`

The synthetic dataset generates metal-like surfaces and injects defects such as scratches, cracks, holes, stains, and dents. This makes the project runnable end to end even when no external benchmark dataset is available. In a real inspection setting, the user can replace these files with actual captured production images.

The design assumption is that good and defective examples should be similar in framing, magnification, and lighting to eventual inference inputs. This matters because both the SVM branch and the R-CNN branch are sensitive to domain mismatch, especially when training data and deployment images differ in background or acquisition geometry.

### 2. Preprocessing Pipeline

The preprocessing logic is implemented in [src/preprocess.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/preprocess.py). The pipeline emphasizes stable defect visibility under practical imaging conditions.

The steps are:

1. Load the image from path, PIL object, or NumPy array.
2. Focus on the main foreground object using threshold-driven contour cropping.
3. Resize to `224 x 224`.
4. Convert to grayscale.
5. Apply CLAHE-based local contrast normalization.
6. Apply Gaussian blur to suppress fine noise.
7. Generate an adaptive threshold map for local defect separation.
8. Compute Sobel edges to highlight structural discontinuities.
9. Apply morphological closing and opening to clean defect masks.

The object-focus stage is especially important because industrial uploads often contain varying borders or surrounding background. By cropping around the primary object, the system reduces background mismatch between training and inference.

### 3. Feature Extraction For The Classical Path

The SVM branch depends on a 58-dimensional feature representation implemented in [src/feature_extract.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/feature_extract.py). This representation is deliberately mixed rather than single-source.

The feature groups are:

- Shape features from extracted contours, including area, perimeter, aspect ratio, solidity, extent, and Hu moments
- Texture features using uniform LBP histograms
- Edge-based features measuring gradient statistics and overlap with defect-mask occupancy
- Statistical grayscale features including mean, standard deviation, higher-order moments, percentiles, and entropy

The rationale is that different defect categories manifest differently. A crack may appear mainly as a contour discontinuity, a stain as an intensity-distribution shift, and roughness as a texture anomaly. Combining descriptors makes the model less dependent on any one failure mode.

### 4. Classical Model Training

The classical training pipeline is implemented in [src/train.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/train.py). The methodology is closer to anomaly detection than ordinary binary classification.

Training procedure:

1. Load both good and defective data from the training split.
2. Fit a `StandardScaler` using only good samples.
3. Train `OneClassSVM` only on good samples.
4. Compute anomaly scores over the labeled training data.
5. Calibrate a decision threshold that maximizes weighted F1.
6. Apply a conservative threshold guardrail based on the upper percentile of good-sample scores to reduce false alarms.
7. Run 5-fold stratified cross-validation.
8. Evaluate on the held-out test split when available.
9. Save the trained detector, scaler, and generated result figures.

This training method reflects a practical manufacturing assumption: good parts define the normal operating state, while defect types may be incomplete or evolving. The threshold calibration stage improves operational usability because raw anomaly scores are difficult to interpret consistently without scaling them into a decision boundary.

### 5. Deep-Learning Model Training

The R-CNN pipeline is implemented in [src/train_rcnn.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/train_rcnn.py) and [src/rcnn_model.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/src/rcnn_model.py).

Key design choices:

- Use Faster R-CNN with a MobileNet-V3-Large-FPN backbone
- Freeze backbone parameters during training to reduce compute load
- Limit the number of training samples for faster CPU-friendly training
- Use a small input size of `224 x 224`
- Auto-generate pseudo bounding boxes for defective images using contour detection on preprocessed masks

For defective samples, contours extracted from cleaned masks are converted into bounding boxes. For good samples, targets contain no boxes. During training, only samples with valid defect boxes contribute to detection loss. This is a compromise between pure classification and fully supervised detection: it adds localization capability while avoiding a manual annotation pipeline.

Evaluation is performed at image level. If the detector outputs any box above the confidence threshold, the image is marked as defective. This simplifies interpretation in the application, where the operator first needs to know whether the part passes and then, if not, where suspect regions appear.

### 6. Inference And Application Workflow

The user-facing application is implemented in [app.py](/abs/path/c:/Users/acer/Downloads/defect%20detection%20v2/defect_detection/app.py). The methodology here is not only predictive but also operational:

- The sidebar controls dataset generation, model choice, and training
- The `Inspect` tab performs single-image inference
- The `Video Analysis` tab applies frame-sampled inference across videos
- The `Training Results` tab exposes model metrics and diagnostic figures
- The `Pipeline Viewer` tab improves interpretability by showing preprocessing stages
- The `Batch History` tab preserves session-level inspection traces

This interface matters methodologically because it closes the loop from data to model to visualization to decision support. The project is not merely a model training repository; it is an inspection workflow prototype.

## Results and Conclusion

### Results

The exact numeric results in this project depend on the dataset used, the selected model family, and whether the training data is synthetic or real. The codebase is structured to expose the following result categories directly:

- Training accuracy
- Weighted F1 score
- Held-out test accuracy
- Held-out test F1 score
- Cross-validation scores for the classical path
- Confusion matrices for both model families
- Classification reports
- Loss curves for the R-CNN path
- Defect probability distributions for image inference
- Defect timeline and per-frame analysis for video inference

From a systems perspective, the results are meaningful in two different ways.

For the `Classic (SVM)` branch:

- The system provides a fast anomaly baseline.
- It works well when good samples are representative and defects create measurable departures in texture, edges, or contour structure.
- It is computationally cheaper and easier to retrain.
- It is especially useful in cases where defect classes are incomplete or evolving.

For the `R-CNN (Deep Learning)` branch:

- The system provides region-level outputs in addition to image-level classification.
- It is more aligned with operator-facing inspection because defect locations are visible.
- It better supports future extension toward richer defect taxonomy and localization tasks.
- It is slower and heavier, but more expressive.

The project also produces practical outputs beyond scalar metrics. Annotated images, annotated videos, confidence charts, and pipeline stages allow the user to inspect whether predictions are plausible, not merely whether the final class label is correct.

### Discussion

Several implementation choices influence result quality:

- Domain alignment is critical. If real inspection inputs differ sharply from the training set, especially in lighting or scale, both branches degrade.
- The SVM branch depends heavily on preprocessing stability because its features are engineered from those processed representations.
- The R-CNN branch depends on the quality of automatically generated pseudo-boxes. These boxes are convenient, but imperfect supervision can limit localization quality.
- Synthetic data is useful for demonstration and debugging, but it is not a substitute for real production data when evaluating deployment readiness.

The architecture therefore fits best as a prototype or academic project that can later be specialized to a true industrial dataset.

### Conclusion

`DefectVision` demonstrates a complete and practical defect-detection system that bridges traditional computer vision and modern deep learning. The classical SVM path shows that well-designed preprocessing and handcrafted descriptors still offer strong value for industrial anomaly detection, especially when computational efficiency and limited labeled data matter. The Faster R-CNN path extends the system with localization capability and a more modern detection-oriented workflow.

The main contribution of this project is not only model performance, but system design. It integrates data generation, training, evaluation, explanation, visualization, and deployment into one coherent application. This makes it useful for teaching, experimentation, benchmarking, and small-scale inspection prototyping.

For future improvement, the most promising directions are:

- Replacing synthetic data with real, domain-specific inspection data
- Using manually verified defect boxes for stronger detector supervision
- Adding precision, recall, ROC, and PR-curve reporting
- Introducing segmentation-based methods for fine-grained defect boundaries
- Packaging the workflow for repeatable deployment or plant-floor integration

Overall, the project shows that defect inspection can be approached effectively with both interpretable classical methods and modern deep-learning methods, and that combining the two in one workflow provides a strong foundation for further development.

## References

The following references are the most relevant conceptual and technical foundations for this project. They cover edge-based image processing, texture representation, SVM-based learning, anomaly detection, object detection, and industrial anomaly benchmarks.

1. N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.

2. J. Canny, “A Computational Approach to Edge Detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8, no. 6, pp. 679–698, 1986.

3. T. Ojala, M. Pietikainen, and T. Maenpaa, “Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 24, no. 7, pp. 971–987, 2002.

4. C. Cortes and V. Vapnik, “Support-Vector Networks,” *Machine Learning*, vol. 20, pp. 273–297, 1995.

5. B. Scholkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, “Estimating the Support of a High-Dimensional Distribution,” *Neural Computation*, vol. 13, no. 7, pp. 1443–1471, 2001.

6. R. Girshick, “Fast R-CNN,” in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2015.

7. S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 39, no. 6, pp. 1137–1149, 2017.

8. A. Howard et al., “Searching for MobileNetV3,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.

9. P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

10. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.

11. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022.

12. Scikit-learn Developers, “scikit-learn: Machine Learning in Python,” software documentation. Relevant components used in this project include `OneClassSVM`, `StandardScaler`, cross-validation utilities, confusion matrices, and classification reports.

13. PyTorch Developers, “PyTorch” and “Torchvision” software documentation. Relevant components used in this project include Faster R-CNN implementations, tensor operations, and model checkpointing.

14. Streamlit Developers, “Streamlit” software documentation. The project uses Streamlit for interactive model control, inspection reporting, and result visualization.
