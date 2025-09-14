# extract_frames.py
import cv2, os, argparse, glob, random
from pathlib import Path

def sample_indices(n_frames, k):
    if n_frames <= k: return list(range(n_frames))
    step = n_frames / (k + 1)
    return [max(0, min(n_frames - 1, int((i+1)*step))) for i in range(k)]

def largest_face(faces):
    if len(faces)==0: return None
    return max(faces, key=lambda f: f[2]*f[3])  # x,y,w,h

def save_face_crop(frame, face, out_path, target=224, pad=0.2):
    h, w = frame.shape[:2]
    if face is None:
        # fallback: center square crop
        s = min(h,w); y0=(h-s)//2; x0=(w-s)//2; crop=frame[y0:y0+s, x0:x0+s]
    else:
        x,y,wf,hf = face
        # add padding around face
        cx, cy = x + wf//2, y + hf//2
        half = int(max(wf,hf)*(0.5+pad))
        x0, y0 = max(0,cx-half), max(0,cy-half)
        x1, y1 = min(w, cx+half), min(h, cy+half)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            s = min(h,w); y0=(h-s)//2; x0=(w-s)//2; crop=frame[y0:y0+s, x0:x0+s]
    crop = cv2.resize(crop, (target,target))
    cv2.imwrite(str(out_path), crop)

def process_dir(src_dir, dst_root, label, frames_per_video=8, train_split=0.8, seed=42):
    random.seed(seed)
    videos = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
    if not videos: print(f"No videos in {src_dir}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    for vid in videos:
        vid_id = Path(vid).stem
        subset = "train" if random.random() < train_split else "val"
        out_dir = Path(dst_root)/subset/label/vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(vid)
        if not cap.isOpened(): 
            print("Skip unreadable:", vid); continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = sample_indices(n_frames, frames_per_video) if n_frames>0 else []

        for i, target_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok, frame = cap.read()
            if not ok: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))
            face = largest_face(faces)
            out_path = out_dir / f"f{i:02d}.jpg"
            save_face_crop(frame, face, out_path)
        cap.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with .mp4 files")
    ap.add_argument("--dst", required=True, help="Output root for frames")
    ap.add_argument("--label", required=True, choices=["real","fake"])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--split", type=float, default=0.8)
    args = ap.parse_args()

    process_dir(args.src, args.dst, args.label, frames_per_video=args.frames, train_split=args.split)
