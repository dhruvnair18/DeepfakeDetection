# backend/inference/video_reader.py
import cv2
import numpy as np
from PIL import Image

def sample_video_frames(file_bytes: bytes, max_frames: int = 16):
    """Returns a list of PIL Images sampled uniformly from the video bytes."""
    # Write to a temp buffer OpenCV can read (OpenCV doesn't read from memory easily)
    # If you prefer, write to a NamedTemporaryFile; here we do simple np buffer trick.
    # However, cv2.VideoCapture requires a path. We'll use NamedTemporaryFile for reliability.
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_count == 0:
            return []

        idxs = np.linspace(0, frame_count - 1, num=min(max_frames, frame_count)).astype(int)

        frames = []
        for target in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
