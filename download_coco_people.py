"""Download three shards of MS-COCO with object annotations (detection-datasets/coco on
Hugging Face, ~1.45 GB). build_coco_people.py then selects photos with a prominent person.

    python download_coco_people.py        (resumable)
"""
import time
from pathlib import Path

from download_v2 import download

OUT = Path(__file__).parent / "data" / "v2" / "coco_ann"
REPO = "https://huggingface.co/datasets/detection-datasets/coco/resolve/main/data/"
SHARDS = ["train-00000-of-00040-67e35002d152155c.parquet", "train-00001-of-00040-2c2b33b9504aa843.parquet",
          "train-00002-of-00040-58e30306870b1de7.parquet"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SHARDS:
        out = OUT / name
        if out.exists():
            print(f"exists {name}", flush=True)
            continue
        t = time.time()
        try:
            download(REPO + name, out)
            print(f"OK   {name} {out.stat().st_size / 1e6:.0f} MB in {(time.time() - t) / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
