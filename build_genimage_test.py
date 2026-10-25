"""Objective O2: build the unseen-generator test set from GenImage.

AI images:   one parquet shard per generator from the BitMind Hugging Face mirror of GenImage
             (data/genimage/hf/<Generator>.parquet). None of these generators were used for
             training (CIFAKE fakes come from Stable Diffusion v1.4).
Real images: ImageNet photos from the GenImage val/nature folders, read directly out of the
             split-archive index parts (data/genimage/raw/*.zip) without the other 3 GB parts.

    python build_genimage_test.py --per-gen 300

Output: data/genimage_test.npz with 32x32 images in CIFAKE format (center crop, bicubic
downscale, JPEG quality 75 - the format CIFAKE uses), labels (1 = AI), generator names, plus
256 px previews in data/genimage/test/<generator>/ for the Streamlit app.
"""
import argparse
import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from genimage_index import read_index

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "genimage" / "raw"
HF = ROOT / "data" / "genimage" / "hf"
PREVIEW = ROOT / "data" / "genimage" / "test"
INDEX_PARTS = ["imagenet_ai_0508_adm.zip", "imagenet_glide.zip"]  # hold the real val/nature photos


def extract(entry, part):
    """Read one file's bytes from a split-archive part, or None if it crosses into another part."""
    with open(part, "rb") as f:
        f.seek(entry["offset"])
        head = f.read(30)
        if head[:4] != b"PK\x03\x04":
            return None
        nlen, xlen = struct.unpack("<HH", head[26:30])
        start = entry["offset"] + 30 + nlen + xlen
        if start + entry["csize"] > part.stat().st_size:
            return None  # data continues in another part
        f.seek(start)
        data = f.read(entry["csize"])
    if entry["method"] == 8:
        return zlib.decompressobj(-15).decompress(data)
    return data if entry["method"] == 0 else None


def to_cifake_format(img):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((32, 32), Image.BICUBIC, reducing_gap=3.0)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def save_preview(img, path):
    img = img.convert("RGB")
    img.thumbnail((256, 256))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90)


def real_images():
    """Yield (name, PIL image) for every ImageNet val/nature photo inside the index parts."""
    seen = set()
    for last in INDEX_PARTS:
        part = RAW / last
        if not part.exists():
            continue
        entries, n_parts = read_index(part)
        for e in entries:
            name = Path(e["name"]).name
            if "/val/nature/" not in e["name"] or e["name"].endswith("/") or e["disk"] != n_parts - 1 or name in seen:
                continue
            raw = extract(e, part)
            if raw is None:
                continue
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
            except Exception:
                continue
            seen.add(name)
            yield name, img


def ai_images(parquet, n, rng):
    """Yield n random (name, PIL image) from one parquet shard, reading only the row groups needed."""
    pf = pq.ParquetFile(parquet)
    col = "image" if "image" in pf.schema_arrow.names else pf.schema_arrow.names[0]
    got = 0
    for rg in rng.permutation(pf.num_row_groups):
        table = pf.read_row_group(int(rg), columns=[col])
        rows = table.column(col).to_pylist()
        for i in rng.permutation(len(rows)):
            cell = rows[i]
            data = cell["bytes"] if isinstance(cell, dict) else cell
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            name = (cell.get("path") if isinstance(cell, dict) else None) or f"{parquet.stem}_{rg}_{i}.png"
            yield Path(name).name, img
            got += 1
            if got == n:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gen", type=int, default=300, help="AI images per generator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    xs, ys, gens, names = [], [], [], []
    for name, img in real_images():
        xs.append(to_cifake_format(img))
        ys.append(0)
        gens.append("Real (ImageNet)")
        names.append(name)
        save_preview(img, PREVIEW / "Real" / (Path(name).stem + ".jpg"))
    print(f"real (ImageNet): {len(ys)} images")

    for parquet in sorted(HF.glob("*.parquet")):
        gen = parquet.stem
        n0 = len(ys)
        for name, img in ai_images(parquet, args.per_gen, rng):
            xs.append(to_cifake_format(img))
            ys.append(1)
            gens.append(gen)
            names.append(name)
            save_preview(img, PREVIEW / gen / (Path(name).stem + ".jpg"))
        print(f"{gen:12s}: {len(ys) - n0} AI images")

    out = ROOT / "data" / "genimage_test.npz"
    np.savez(out, x=np.stack(xs), y=np.array(ys), gen=np.array(gens), name=np.array(names))
    print(f"saved {out.name}: {len(ys)} images ({sum(ys)} AI, {len(ys) - sum(ys)} real)")


if __name__ == "__main__":
    main()
