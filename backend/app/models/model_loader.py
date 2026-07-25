"""
model_loader.py

Loads BOTH models once at startup and keeps them in memory:
  1. A general-purpose ResNet18 pretrained on ImageNet (1000 everyday
     object categories) -- used for ordinary photos.
  2. A fine-tuned ResNet18 for brain tumor MRI classification (glioma /
     meningioma / pituitary / notumor) -- used when the upload looks
     like a grayscale medical scan.

predict.py picks which one to use per-request based on a simple
image heuristic (see routes/predict.py).

The tumor weights file (~45MB) is too big for a normal GitHub file
upload (25MB limit), so it's hosted as a GitHub Release asset and
downloaded automatically the first time the server starts.
"""

import json
import os

import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_TUMOR_WEIGHTS_PATH = os.path.join(_MODELS_DIR, "brain_tumor_resnet18.pth")
_TUMOR_CLASS_NAMES_PATH = os.path.join(_MODELS_DIR, "class_names.json")

_TUMOR_WEIGHTS_URL = (
    "https://github.com/samm67363-ui/ai-image/releases/download/"
    "model-v1/brain_tumor_resnet18.pth"
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_tumor_weights_if_missing():
    if os.path.exists(_TUMOR_WEIGHTS_PATH):
        return
    print(f"Downloading model weights from {_TUMOR_WEIGHTS_URL} ...")
    response = requests.get(_TUMOR_WEIGHTS_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(_TUMOR_WEIGHTS_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model weights downloaded successfully.")


# ---------------------------------------------------------------
# General-purpose model (everyday photos, 1000 ImageNet classes)
# ---------------------------------------------------------------
_general_weights = ResNet18_Weights.DEFAULT
_general_model = resnet18(weights=_general_weights)
_general_model.eval()
_general_model.to(_device)
_general_class_names = _general_weights.meta["categories"]


# ---------------------------------------------------------------
# Fine-tuned brain-tumor model (4 classes)
# ---------------------------------------------------------------
_download_tumor_weights_if_missing()

if not os.path.exists(_TUMOR_CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "Missing class_names.json in backend/app/models/. "
        "This is produced alongside the .pth file by the Kaggle "
        "training script -- upload it there directly (it's tiny, well "
        "under GitHub's normal upload limit)."
    )

with open(_TUMOR_CLASS_NAMES_PATH) as f:
    _tumor_class_names = json.load(f)

_tumor_model = resnet18(weights=None)
_tumor_model.fc = nn.Linear(_tumor_model.fc.in_features, len(_tumor_class_names))
_tumor_model.load_state_dict(torch.load(_TUMOR_WEIGHTS_PATH, map_location=_device))
_tumor_model.eval()
_tumor_model.to(_device)


def get_general_model():
    return _general_model


def get_general_class_names():
    return _general_class_names


def get_tumor_model():
    return _tumor_model


def get_tumor_class_names():
    return _tumor_class_names


def get_device():
    return _device
