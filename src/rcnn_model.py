"""
rcnn_model.py
-------------
Faster R-CNN model builder for industrial defect detection.
Uses a lightweight MobileNet-V3-Large backbone for fast CPU training,
or optionally ResNet-50-FPN for GPU-accelerated training.
"""

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 2  # background + defect

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_rcnn_model(pretrained_backbone: bool = True, freeze_backbone: bool = True) -> torch.nn.Module:
    """
    Build a Faster R-CNN model with a MobileNet-V3-Large-FPN backbone.
    This is ~5x faster than ResNet-50 on CPU while still accurate.

    Args:
        pretrained_backbone: If True, use ImageNet pre-trained weights (transfer learning).
        freeze_backbone: If True, freeze backbone weights for faster training.

    Returns:
        A Faster R-CNN model ready for training or inference.
    """
    if pretrained_backbone:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    # Replace the classification head for our 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Freeze backbone for much faster CPU training (only train the head)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def get_device() -> torch.device:
    """Return the best available device."""
    return DEVICE


def model_summary(model: torch.nn.Module) -> dict:
    """Return a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(DEVICE),
        "backbone": "MobileNet-V3-Large-FPN",
        "num_classes": NUM_CLASSES,
    }
