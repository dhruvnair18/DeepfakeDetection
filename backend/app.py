# backend/app.py
import os
import io
import time
from typing import List, Literal, Optional

# Load .env first so env vars are available to everything below
from dotenv import load_dotenv
load_dotenv()
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
FAKE_INDEX = int(os.getenv("FAKE_INDEX", "0"))

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

from backend.inference.loader import ModelLoader
from backend.inference.preprocess import get_transform
from backend.inference.video_reader import sample_video_frames
from backend.db import SessionLocal
from backend.models import InferenceResult, init_db

app = FastAPI(title="Deepfake Detector API", version="0.1.0")
init_db()

transform = get_transform()
model = ModelLoader.get()

class PredictResponse(BaseModel):
    label: Literal["real", "fake"]
    probability: float
    n_frames_used: int
    frame_scores: Optional[List[float]] = None
    model_version: str
    inference_time_ms: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "api_version": app.version,
        "model_version": model.version,
        "threshold": THRESHOLD,
        "fake_index": FAKE_INDEX,
    }

@app.get("/history")
def history(limit: int = 10):
    db = SessionLocal()
    try:
        rows = db.query(InferenceResult).order_by(InferenceResult.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "media_type": r.media_type,
                "label": r.label,
                "probability": r.probability,
                "n_frames_used": r.n_frames_used,
                "model_version": r.model_version,
                "created_at": r.created_at.isoformat() + "Z",
            }
            for r in rows
        ]
    finally:
        db.close()

def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    # logits: [B, 2] -> np array
    x = logits.numpy()
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    filename = file.filename or "upload"
    content = await file.read()

    # Safe content-type handling
    ctype = (file.content_type or "").lower()
    is_image = ctype.startswith("image/")
    is_video = ctype.startswith("video/") or filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

    if not (is_image or is_video):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    # Build batch of tensors
    images = []
    media_type = "image"
    if is_image:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot read image")
        images = [img]
    else:
        media_type = "video"
        frames = sample_video_frames(content, max_frames=16)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode video or no frames found")
        images = frames

    try:
        batch = torch.stack([transform(im) for im in images], dim=0)  # [N,3,224,224]
    except Exception:
        raise HTTPException(status_code=400, detail="Preprocessing failed")

    # Inference
    logits = model.predict_logits(batch)         # [N,2] on CPU
    probs = _softmax_np(logits)                  # [N,2]
    frame_fake_probs = probs[:, FAKE_INDEX].tolist()
    mean_fake = float(np.mean(frame_fake_probs))

    label = "fake" if mean_fake >= THRESHOLD else "real"
    prob = mean_fake if label == "fake" else 1.0 - mean_fake

    # Persist
    db = SessionLocal()
    try:
        row = InferenceResult(
            filename=filename,
            media_type=media_type,
            label=label,
            probability=prob,
            n_frames_used=len(images),
            model_version=model.version,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()

    dt_ms = int((time.time() - t0) * 1000)
    return PredictResponse(
        label=label,
        probability=round(prob, 4),
        n_frames_used=len(images),
        frame_scores=[round(s, 4) for s in frame_fake_probs] if media_type == "video" else None,
        model_version=model.version,
        inference_time_ms=dt_ms,
    )
