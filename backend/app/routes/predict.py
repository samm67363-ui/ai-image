"""
predict.py

Defines the POST /predict endpoint:
  1. Accepts an uploaded image file.
  2. Validates it.
  3. Preprocesses it into a tensor.
  4. Runs it through the pretrained ResNet18 model.
  5. Returns the top-1 class + confidence, and the top-5 predictions.
"""

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.model_loader import get_model, get_weights, get_device
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
    weights = get_weights()
    try:
        input_tensor = preprocess_image(file_bytes, weights)
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
            output = model(input_tensor)  # raw logits, shape [1, 1000]
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        )

    # ---- Step 5: get top 5 predictions ----
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    categories = weights.meta["categories"]  # list of 1000 ImageNet class names

    top5 = []
    for prob, idx in zip(top5_prob, top5_indices):
        top5.append(
            {
                "class": categories[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
        )

    return {
        "class": top5[0]["class"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }
