"""Portrait safeguard: find faces so the app can avoid confident verdicts on portraits.

The v2 detector is unreliable on photos of people at medium distance: it flagged real phone
portraits as AI and passed a photorealistic AI portrait as real (see README). find_faces() uses
OpenCV's YuNet face detector (models/face_detection_yunet_2023mar.onnx, from opencv_zoo, MIT
license); portrait_zone() says whether the largest face falls in the size range where the model
was measured to be unreliable.

    from portrait import portrait_zone
    zone = portrait_zone(pil_image)   # None, or {"faces": n, "face_size": 0.18, "in_zone": True}
"""
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
# Face width as a share of the image's shorter side. Set from the measurement in the README
# (evaluate with: python portrait.py --evaluate).
ZONE = (0.08, 0.55)
_detector = None


def find_faces(img, max_side=640, min_conf=0.8):
    """Faces in a PIL image as a list of (x, y, w, h, confidence) in pixels of the resized image,
    plus that image's (width, height)."""
    global _detector
    import cv2
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    a = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), min_conf, 0.3, 5000)
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(a)
    faces = [] if faces is None else [tuple(float(v) for v in f[:4]) + (float(f[14]),) for f in faces]
    return faces, (w, h)


def portrait_zone(img):
    """None if no face was found, else the largest face's size and whether it is in ZONE."""
    try:
        faces, (w, h) = find_faces(img)
    except Exception:
        return None  # OpenCV or the model file missing: no safeguard rather than a crash
    if not faces:
        return None
    size = max(f[2] for f in faces) / min(w, h)
    return {"faces": len(faces), "face_size": size, "in_zone": ZONE[0] <= size <= ZONE[1]}


def evaluate():
    """Model accuracy on the v2 test and held-out images, grouped by the size of the largest face."""
    import torch
    from PIL import Image
    from predict import load_model, newest_run
    from train_v2 import V2, predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = np.load(V2 / "images.npy", mmap_mode="r")
    meta = dict(np.load(V2 / "meta.npz"))
    idx = np.flatnonzero(np.isin(meta["split"], ["test", "unseen"]))
    p = predict(load_model(newest_run(), device), images, idx, device)
    sizes = []
    for i in np.sort(idx):
        faces, (w, h) = find_faces(Image.fromarray(np.asarray(images[i])), max_side=224)
        sizes.append(max(f[2] for f in faces) / min(w, h) if faces else 0.0)
    sizes, y = np.array(sizes), meta["label"][np.sort(idx)]
    correct = (p >= 0.5) == (y == 1)
    bins = [(0, 0, "no face"), (1e-9, 0.08, "< 8%"), (0.08, 0.20, "8-20%"), (0.20, 0.35, "20-35%"),
            (0.35, 0.55, "35-55%"), (0.55, 9, "> 55%")]
    print(f"{'largest face (share of shorter side)':38s}{'images':>8s}{'real':>7s}{'AI':>6s}"
          f"{'real correct':>14s}{'AI correct':>12s}{'balanced':>10s}")
    for lo, hi, label in bins:
        m = (sizes == 0) if hi == 0 else (sizes >= lo) & (sizes < hi)
        r, a = m & (y == 0), m & (y == 1)
        rc = correct[r].mean() if r.any() else float("nan")
        ac = correct[a].mean() if a.any() else float("nan")
        print(f"{label:38s}{m.sum():8d}{r.sum():7d}{a.sum():6d}{100 * rc:13.1f}%{100 * ac:11.1f}%{50 * (rc + ac):9.1f}%")
    zone = (sizes >= ZONE[0]) & (sizes <= ZONE[1])
    print(f"\nsafeguard zone {ZONE[0]:.0%}-{ZONE[1]:.0%}: fires on {zone.mean():.1%} of these images; "
          f"accuracy inside {correct[zone].mean():.1%} vs outside {correct[~zone].mean():.1%}")
    for s in sorted(set(meta["source"][np.sort(idx)])):
        k = meta["source"][np.sort(idx)] == s
        print(f"   {s:26s} fires on {zone[k].mean():5.1%} of {k.sum()} images")


if __name__ == "__main__":
    import sys
    if "--evaluate" in sys.argv:
        evaluate()
