"""Build the v2 (224x224) dataset for a detector that works on large, real-world images.

Sources (one parquet shard each; see download_v2.py and build_genimage_test.py):
  real:  FFHQ faces, CelebA-HQ faces, MS-COCO photos            (+ ImageNet photos, test only)
  AI:    SDXL faces, SDXL scenes, RealVisXL, JourneyDB (Midjourney), GenImage ADM / VQDM / Wukong
  AI held out of training completely ("unseen" tests): FLUX faces, FLUX scenes,
         GenImage GLIDE, GenImage Midjourney, GenImage BigGAN

Every image, real or AI, gets identical processing: RGB, shorter side resized to 224 (bicubic,
antialiased), center crop 224x224. Output:
  data/v2/images.u8   uint8 memmap [N, 224, 224, 3]
  data/v2/meta.npz    label (1 = AI), source, split (train / val / test / unseen)

    python build_v2_dataset.py
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from build_genimage_test import real_images

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "v2"
SIZE = 224
UNSEEN = {"ai_ffhq_flux", "ai_coco_flux", "ai_genimage_GLIDE", "ai_genimage_Midjourney", "ai_genimage_BigGAN",
          "real_camera",
          "real_imagenet"}


def to_v2_format(img, size=SIZE):
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)))


def parquet_images(path, n, rng):
    """Yield up to n random decoded PIL images from a parquet shard (reads only the row groups needed)."""
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    col = "image" if "image" in names else next(c for c in names if "image" in c.lower() or "img" in c.lower())
    got = 0
    for rg in rng.permutation(pf.num_row_groups):
        rows = pf.read_row_group(int(rg), columns=[col]).column(col).to_pylist()
        for i in rng.permutation(len(rows)):
            cell = rows[i]
            data = cell["bytes"] if isinstance(cell, dict) else cell
            if not data:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            yield img
            got += 1
            if got == n:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-cap", type=int, default=4500, help="max images per real source")
    ap.add_argument("--ai-cap", type=int, default=2000, help="max images per AI source")
    ap.add_argument("--unseen-cap", type=int, default=1000, help="max images per held-out source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sources = {p.stem: p for p in sorted((V2 / "raw").glob("*.parquet"))}
    sources.update({f"ai_genimage_{p.stem}": p for p in sorted((ROOT / "data" / "genimage" / "hf").glob("*.parquet"))})
    plan = []
    for name, path in sources.items():
        cap = args.unseen_cap if name in UNSEEN else (args.real_cap if name.startswith("real_") else args.ai_cap)
        plan.append((name, path, cap))
    n_max = sum(c for _, _, c in plan) + 1200  # + ImageNet photos

    V2.mkdir(parents=True, exist_ok=True)
    tmp = np.lib.format.open_memmap(V2 / "images_tmp.npy", mode="w+", dtype=np.uint8, shape=(n_max, SIZE, SIZE, 3))
    labels, srcs, native = [], [], []
    n = 0

    def add(img, name):
        nonlocal n
        tmp[n] = to_v2_format(img)
        labels.append(0 if name.startswith("real_") else 1)
        srcs.append(name)
        native.append(min(img.size))
        n += 1

    for name, path, cap in plan:
        n0 = n
        for img in parquet_images(path, cap, rng):
            add(img, name)
        sizes = np.array(native[n0:])
        print(f"{name:26s} {n - n0:5d} images | native shorter side: median {int(np.median(sizes)) if len(sizes) else 0} px"
              f"{'  [held out: unseen]' if name in UNSEEN else ''}", flush=True)
    n0 = n
    for _, img in real_images():
        add(img, "real_imagenet")
    print(f"{'real_imagenet':26s} {n - n0:5d} images  [held out: unseen]", flush=True)

    labels, srcs = np.array(labels), np.array(srcs)
    split = np.empty(n, dtype="U6")
    for s in np.unique(srcs):
        idx = rng.permutation(np.flatnonzero(srcs == s))
        if s in UNSEEN:
            split[idx] = "unseen"
        else:
            a, b = int(0.85 * len(idx)), int(0.90 * len(idx))
            split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"

    final = np.lib.format.open_memmap(V2 / "images.npy", mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    for i in range(0, n, 2000):
        j = min(i + 2000, n)  # tmp is allocated larger than n
        final[i:j] = tmp[i:j]
    final.flush()
    del tmp, final
    (V2 / "images_tmp.npy").unlink()
    np.savez(V2 / "meta.npz", label=labels, source=srcs, split=split, native=np.array(native))
    print(f"\nsaved data/v2/images.npy: {n} images at {SIZE}x{SIZE}")
    for sp in ["train", "val", "test", "unseen"]:
        m = split == sp
        print(f"  {sp:6s} {int(m.sum()):6d}  (real {int((labels[m] == 0).sum())}, AI {int((labels[m] == 1).sum())})")


if __name__ == "__main__":
    main()
