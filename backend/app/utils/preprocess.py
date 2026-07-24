"""
preprocess.py

Handles turning raw uploaded image bytes into a normalized tensor
that the fine-tuned brain-tumor ResNet18 expects: resize to 224x224,
convert to tensor, normalize with the same ImageNet statistics used
during training (see training/kaggle_train_brain_tumor_model.py).
"""

import io
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

# Max upload size we accept: 8 MB. Anything larger is rejected
# before we even try to decode it, to avoid memory/DoS issues.
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024

# Allowed image MIME types.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class InvalidImageError(Exception):
    """Raised when the uploaded file is not a usable image."""
    pass


def validate_upload(content_type: str, file_bytes: bytes) -> None:
    """
    Raises InvalidImageError if the upload fails basic checks:
    - too large
    - not an allowed content type
    - not actually decodable as an image
    """
    if len(file_bytes) == 0:
        raise InvalidImageError("Uploaded file is empty.")

    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise InvalidImageError(
            f"Image too large. Max allowed size is "
            f"{MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported file type '{content_type}'. "
            f"Allowed types: JPEG, PNG, WEBP, BMP."
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # checks the file is a genuine, non-corrupted image
    except (UnidentifiedImageError, OSError):
        raise InvalidImageError("File could not be read as a valid image.")


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Converts raw image bytes into a batched tensor ready for the model:
    resize -> tensor -> normalize -> add batch dimension.
    """
    # Re-open the image (verify() above consumes the file pointer,
    # so we open it fresh here rather than reusing the same buffer).
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    tensor = _transform(image)  # shape: [3, 224, 224]

    # Add a batch dimension -> [1, 3, 224, 224], since the model
    # always expects a batch, even if it's a batch of one image.
    batched_tensor = tensor.unsqueeze(0)

    return batched_tensor
