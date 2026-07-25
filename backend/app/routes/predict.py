"""
predict.py

Defines the POST /predict endpoint. Auto-detects whether the upload
looks like a grayscale medical scan (MRI) or an ordinary color photo,
and routes it to the matching model:
  - Looks like MRI -> fine-tuned brain-tumor model (4 classes)
  - Otherwise       -> general ImageNet model (1000 classes)

Returns which mode was used, the top-1 class + confidence, and either
the full ranked list (MRI mode, 4 classes) or the top 5 (general mode,
1000 classes).

Educational project only -- not a medical device. Never use tumor-mode
output for real diagnosis or treatment decisions.
"""

import torch
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.model_loader import (
    get_general_model,
    get_general_class_names,
    get_tumor_model,
    get_tumor_class_names,
    get_device,
)
from app.utils.preprocess import (
    validate_upload,
    preprocess_image,
    looks_like_mri,
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
        raise HTTPException(status_code=400, detail=str(e))

    # ---- Step 3: decide which model to use ----
    mri_mode = looks_like_mri(file_bytes)

    # ---- Step 4: preprocess into a model-ready tensor ----
    try:
        input_tensor = preprocess_image(file_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to process image. Please try a different file.",
        )

    device = get_device()
    input_tensor = input_tensor.to(device)

    if mri_mode:
        model = get_tumor_model()
        class_names = get_tumor_class_names()
    else:
        model = get_general_model()
        class_names = get_general_class_names()

    # ---- Step 5: run inference ----
    try:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        )

    # ---- Step 6: rank predictions ----
    # MRI mode has only 4 classes total, so return all of them.
    # General mode has 1000, so just return the top 5.
    top_k = len(class_names) if mri_mode else 5
    sorted_probs, sorted_indices = torch.topk(probabilities, top_k)

    ranked = []
    for prob, idx in zip(sorted_probs, sorted_indices):
        ranked.append(
            {
                "class": class_names[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
        )

    return {
        "mode": "mri" if mri_mode else "general",
        "class": ranked[0]["class"],
        "confidence": ranked[0]["confidence"],
        "top5": ranked,
    }
