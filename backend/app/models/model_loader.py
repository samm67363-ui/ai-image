"""
model_loader.py

Loads the fine-tuned brain-tumor ResNet18 model exactly once and keeps
it in memory. FastAPI imports get_model() / get_class_names() /
get_device() from here instead of reloading anything per request.

The weights file (brain_tumor_resnet18.pth) is ~45MB, too big for a
normal GitHub file upload (25MB limit), so it's hosted as a GitHub
Release asset instead and downloaded automatically the first time the
server starts. class_names.json is tiny and lives directly in the repo
alongside this file.
"""

import json
import os

import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_PATH = os.path.join(_MODELS_DIR, "brain_tumor_resnet18.pth")
_CLASS_NAMES_PATH = os.path.join(_MODELS_DIR, "class_names.json")

# Direct download link to the .pth file attached to a GitHub Release.
# Update this if you publish a new release with updated weights.
_WEIGHTS_URL = (
    "https://github.com/samm67363-ui/ai-image/releases/download/"
    "model-v1/brain_tumor_resnet18.pth"
)


def _download_weights_if_missing():
    """
    Downloads the model weights from the GitHub Release into
    _WEIGHTS_PATH if they aren't already there. Runs once per server
    boot -- on Render's free tier the disk is ephemeral, so this
    re-downloads after the service restarts or redeploys, but not on
    every request.
    """
    if os.path.exists(_WEIGHTS_PATH):
        return

    print(f"Downloading model weights from {_WEIGHTS_URL} ...")
    response = requests.get(_WEIGHTS_URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(_WEIGHTS_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Model weights downloaded successfully.")


_download_weights_if_missing()

if not os.path.exists(_CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "Missing class_names.json in backend/app/models/. "
        "This is produced alongside the .pth file by the Kaggle "
        "training script -- upload it there directly (it's tiny, well "
        "under GitHub's normal upload limit)."
    )

with open(_CLASS_NAMES_PATH) as f:
    _class_names = json.load(f)

# Use GPU if available, otherwise fall back to CPU.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild the same architecture used during training: a ResNet18 with
# its final layer swapped from 1000 ImageNet classes to our tumor classes.
_model = resnet18(weights=None)
_model.fc = nn.Linear(_model.fc.in_features, len(_class_names))
_model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=_device))
_model.eval()
_model.to(_device)


def get_model():
    """Return the singleton fine-tuned model, already in eval mode."""
    return _model


def get_class_names():
    """Return the list of class names, in model output order."""
    return _class_names


def get_device():
    """Return the torch device the model is loaded on (cpu or cuda)."""
    return _device
