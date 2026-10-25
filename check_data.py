"""Sanity-check the CIFAKE dataset and save a grid of sample images.

Usage:  python check_data.py
Output: counts per split/class, image size/mode, JPEG quality check, samples.png
"""
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / "data" / "cifake"
SPLITS, CLASSES = ["train", "test"], ["REAL", "FAKE"]
random.seed(0)


def jpeg_quality_signature(path):
    """Return the first value of the luminance quantization table.

    Different values between REAL and FAKE would mean the two classes were saved
    with different JPEG settings - a shortcut the model could learn instead of
    real generation artifacts.
    """
    with Image.open(path) as im:
        q = getattr(im, "quantization", None)
        return q[0][0] if q else None


def main():
    print("Image counts")
    for split in SPLITS:
        for cls in CLASSES:
            n = len(list((ROOT / split / cls).glob("*")))
            print(f"  {split:5s} {cls:4s} {n:>6}")

    print("\nImage format (200 random files per class)")
    for cls in CLASSES:
        files = random.sample(list((ROOT / "train" / cls).glob("*")), 200)
        sizes, modes, quality = Counter(), Counter(), Counter()
        for f in files:
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
            quality[jpeg_quality_signature(f)] += 1
        print(f"  {cls}: sizes={dict(sizes)} modes={dict(modes)} jpeg_q_table[0]={dict(quality)}")

    # 2 rows (REAL, FAKE) x 8 samples, upscaled so 32x32 images are visible
    n, scale, pad, label_w = 8, 4, 4, 60
    tile = 32 * scale
    grid = Image.new("RGB", (label_w + n * (tile + pad), 2 * (tile + pad)), "white")
    draw = ImageDraw.Draw(grid)
    for row, cls in enumerate(CLASSES):
        draw.text((5, row * (tile + pad) + tile // 2), cls, fill="black")
        for col, f in enumerate(random.sample(list((ROOT / "test" / cls).glob("*")), n)):
            with Image.open(f) as im:
                img = im.convert("RGB").resize((tile, tile), Image.NEAREST)
            grid.paste(img, (label_w + col * (tile + pad), row * (tile + pad)))
    out = Path(__file__).parent / "samples.png"
    grid.save(out)
    print(f"\nSaved sample grid to {out}")


if __name__ == "__main__":
    main()
