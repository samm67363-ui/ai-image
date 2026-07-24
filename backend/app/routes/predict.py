"""
predict.py

Defines the POST /predict endpoint:
  1. Accepts an uploaded image file.
  2. Validates it.
  3. Preprocesses it into a tensor.
  4. Runs it through the fine-tuned brain-tumor ResNet18 model.
  5. Returns the top-1 class + confidence, and the full ranked list
     (there are only 4 classes total, so "top5" becomes "all classes").

Educational project only -- not a medical device. Never use this
output for real diagnosis or treatment decisions.
"""

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.model_loader import get_model, get_class_names, get_device
from app.utils.preprocess import (
    validate_upload,
    preprocess_image,
    InvalidImageError,
)

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---- Step 1: read raw bytes from the upload ----
    file_bytes = await file.read()

    # ---- Step 2: validate (type, size, corruption) ----
    try:
        validate_upload(file.content_type, file_bytes)
    except InvalidImageError as e:
        # 400 = client sent something we can't use.
        raise HTTPException(status_code=400, detail=str(e))

    # ---- Step 3: preprocess into a model-ready tensor ----
    try:
        input_tensor = preprocess_image(file_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to process image. Please try a different file.",
        )

    model = get_model()
    device = get_device()
    input_tensor = input_tensor.to(device)

    # ---- Step 4: run inference ----
    try:
        with torch.no_grad():  # no gradient tracking needed at inference time
            output = model(input_tensor)  # raw logits, shape [1, num_classes]
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        )

    # ---- Step 5: rank all classes (only 4 total, so we return all of them) ----
    class_names = get_class_names()
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    ranked = []
    for prob, idx in zip(sorted_probs, sorted_indices):
        ranked.append(
            {
                "class": class_names[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
        )

    return {
        "class": ranked[0]["class"],
        "confidence": ranked[0]["confidence"],
        "top5": ranked,
    }
