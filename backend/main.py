"""
main.py

Entry point for the FastAPI backend. Run with:

    uvicorn main:app --reload

This wires together CORS (so the React frontend on a different
origin can call this API), the health-check route, and the
/predict route defined in app/routes/predict.py.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router

app = FastAPI(
    title="AI Image Classifier API",
    description="Backend API that classifies uploaded images using a pretrained ResNet18 model.",
    version="1.0.0",
)

# ---------------------------------------------------------------
# CORS setup
# ---------------------------------------------------------------
# During local development the React app runs on http://localhost:5173
# (Vite's default port) while this API runs on http://localhost:8000.
# Different ports = different "origins" to the browser, so without
# CORS enabled, the browser blocks the frontend's requests.
#
# In production (frontend on Vercel, backend on Render), set the
# ALLOWED_ORIGINS environment variable on Render to a comma-separated
# list of the exact frontend URLs, e.g.:
#
#   ALLOWED_ORIGINS=https://image-classifier.vercel.app,https://image-classifier-git-main-you.vercel.app
#
# If ALLOWED_ORIGINS isn't set, it falls back to localhost only, and
# any *.vercel.app preview URL is allowed via a regex so preview
# deployments (which get a new URL each time) keep working.
# ---------------------------------------------------------------
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/")
def health_check():
    """Simple health check so you can confirm the server is alive."""
    return {"status": "ok", "message": "AI Image Classifier API is running."}
