"""Download the v2 (high-resolution) training data: one parquet shard per source from the
BitMind Hugging Face datasets. Resumable: re-run to continue interrupted downloads.

    python download_v2.py

Files go to data/v2/raw/<name>.parquet. Names start with real_ or ai_ (the label).
"""
import os
import time
from pathlib import Path

import requests

OUT = Path(__file__).parent / "data" / "v2" / "raw"
SHARDS = {
    "real_ffhq": ("bitmind/ffhq-256", "data/train-00000-of-00016.parquet"),
    "ai_ffhq_sdxl": ("bitmind/ffhq-256___stable-diffusion-xl-base-1.0", "data/train-00000-of-00014.parquet"),
    "real_coco": ("bitmind/MS-COCO-unique-256", "data/train-00000-of-00028.parquet"),
    "ai_coco_sdxl": ("bitmind/MS-COCO-unique___stable-diffusion-xl-base-1.0", "data/train-00000-of-00023.parquet"),
    "ai_realvisxl": ("bitmind/bm-realvisxl", "data/train-00000-of-00026.parquet"),
    "real_celebahq": ("bitmind/celeb-a-hq", "data/train-00000-of-00006.parquet"),
    "ai_journeydb": ("bitmind/JourneyDB", "data/test-00000-of-00009.parquet"),
    "ai_ffhq_flux": ("bitmind/ffhq-256___FLUX.1-dev", "data/train-00000-of-00014.parquet"),
    "ai_coco_flux": ("bitmind/MS-COCO-unique___FLUX.1-dev", "data/train-00000-of-00023.parquet"),
    # Real object-centric photos, added to stop v2 using "object/animal content" as a cue for AI
    "real_caltech256": ("bitmind/caltech-256", "data/train-00000-of-00004.parquet"),
    "real_openimages": ("bitmind/open-image-v7-256", "data/train-00000-of-00432.parquet"),
    # Real people in ordinary compositions (not tight face crops), added to stop v2 using
    # "waist-up portrait with a soft background" as a cue for AI
    "real_lfw": ("bitmind/lfw", "data/train-00000-of-00001.parquet"),
    "real_openimages_hi": ("bitmind/open-images-v7-subset", "data/test-00000-of-00077.parquet"),
    # Full-resolution camera/phone photos: never trained on, used only as a held-out test
    "real_camera": ("bitmind/bm-real", "data/train-00000-of-00045.parquet"),
}


def download(url, out, tries=8):
    part = Path(str(out) + ".part")
    for attempt in range(tries):
        done = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, stream=True, timeout=60, headers={"Range": f"bytes={done}-"} if done else {}) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except Exception as e:  # network hiccup: wait and resume
            print(f"  retry {attempt + 1}/{tries} after {type(e).__name__}", flush=True)
            time.sleep(15)
    else:
        raise RuntimeError("too many failed attempts")
    os.replace(part, out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (repo, path) in SHARDS.items():
        out = OUT / f"{name}.parquet"
        if out.exists():
            print(f"exists {name}", flush=True)
            continue
        t = time.time()
        try:
            download(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", out)
            print(f"OK   {name:14s} {out.stat().st_size / 1e6:5.0f} MB in {(time.time() - t) / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
