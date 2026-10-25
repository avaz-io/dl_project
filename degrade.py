"""Image degradations for robustness experiments (objective O3).

Shared by train.py (random compression-aware augmentation) and robustness.py
(fixed test conditions), so training and evaluation use identical operations.
All functions work on 32x32 RGB uint8 images.
"""
import io

import numpy as np
from PIL import Image, ImageFilter


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rescale(img, scale):
    """Downscale by `scale`, then back up to the original size (loses fine detail)."""
    w, h = img.size
    small = img.resize((max(4, round(w * scale)), max(4, round(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(sigma))


def messenger(img):
    """Rough imitation of a messaging app re-sharing a photo: downscale + JPEG re-compression."""
    return jpeg(rescale(img, 0.75), 70)


# Fixed test conditions. JPEG 30 and scale 0.4 are harsher than anything seen in
# compression-aware training, so they test extrapolation rather than memorization.
CONDITIONS = {
    "clean": None,
    "jpeg90": lambda im: jpeg(im, 90),
    "jpeg75": lambda im: jpeg(im, 75),
    "jpeg50": lambda im: jpeg(im, 50),
    "jpeg30": lambda im: jpeg(im, 30),
    "scale0.75": lambda im: rescale(im, 0.75),
    "scale0.5": lambda im: rescale(im, 0.5),
    "scale0.4": lambda im: rescale(im, 0.4),
    "blur0.5": lambda im: blur(im, 0.5),
    "blur1.0": lambda im: blur(im, 1.0),
    "messenger": messenger,
}


def apply(x_uint8, fn):
    """Apply a PIL-image function to every image in a (N, H, W, 3) uint8 array."""
    if fn is None:
        return x_uint8
    return np.stack([np.asarray(fn(Image.fromarray(a))) for a in x_uint8])


def random_degrade(x_uint8, rng, p=0.5):
    """Compression-aware augmentation: with probability p per image, apply a random
    chain of rescale (0.5-1.0), blur (sigma 0.2-1.0) and JPEG (quality 40-95).
    Applied independently of the label, so both classes are degraded equally."""
    out = x_uint8.copy()
    for i in np.flatnonzero(rng.random(len(x_uint8)) < p):
        img = Image.fromarray(x_uint8[i])
        if rng.random() < 0.5:
            img = rescale(img, rng.uniform(0.5, 1.0))
        if rng.random() < 0.5:
            img = blur(img, rng.uniform(0.2, 1.0))
        if rng.random() < 0.7:
            img = jpeg(img, rng.integers(40, 96))
        out[i] = np.asarray(img)
    return out
