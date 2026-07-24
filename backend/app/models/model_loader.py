"""
model_loader.py

Loads the pretrained ResNet18 model (ImageNet weights) exactly once
and keeps it in memory. FastAPI will import `get_model()` and
`get_weights()` from here instead of reloading the model on every
request (which would be extremely slow).
"""

import torch
from torchvision.models import resnet18, ResNet18_Weights

# ---------------------------------------------------------------
# Load weights + model ONE TIME at import time (module-level).
# Python caches modules after first import, so this code runs
# only once for the lifetime of the server process.
# ---------------------------------------------------------------

_weights = ResNet18_Weights.DEFAULT  # best available pretrained weights
_model = resnet18(weights=_weights)
_model.eval()  # inference mode: disables dropout/batchnorm training behavior

# Use GPU if available, otherwise fall back to CPU.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)


def get_model():
    """Return the singleton ResNet18 model, already in eval mode."""
    return _model


def get_weights():
    """Return the ResNet18_Weights enum (holds transforms + class labels)."""
    return _weights


def get_device():
    """Return the torch device the model is loaded on (cpu or cuda)."""
    return _device
