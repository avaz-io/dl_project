"""Select real MS-COCO photos in which a person is prominent, and write them (plus person-centred
crops) to data/v2/raw/real_coco_people.parquet, which build_v2_dataset.py then picks up as a
real source.

    python download_coco_people.py && python build_coco_people.py

Why: v2 called real photos "AI" as soon as a person's torso was in frame, because every real
person in its training data was a face or head-and-shoulders shot while AI portraits showed
people from the chest or waist up. COCO's person boxes let us pick real photos with exactly
those compositions:
  - the full photo, when the largest person box covers at least MIN_AREA of the image
  - a square crop from the top of that box: chest-up, waist-up or whole person (random extent),
    kept only if the crop is at least MIN_CROP px wide so it is not a blurry upscale
"""
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "v2" / "coco_ann"
OUT = ROOT / "data" / "v2" / "raw" / "real_coco_people.parquet"
MIN_AREA, MIN_HEIGHT, MIN_CROP = 0.12, 0.45, 150


# Category index of "person". The parquet files carry no class names; index 0 is "person" in the
# dataset's published feature list, and 54% of the photos contain it, matching COCO's known figure.
PERSON = 0


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG")  # lossless: no extra JPEG generation on top of COCO's own
    return b.getvalue()


def main():
    rng = np.random.default_rng(0)
    out, stats = [], {"photos": 0, "with_prominent_person": 0, "crops": 0}
    for shard in sorted(SRC.glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        person = PERSON
        for batch in pf.iter_batches(batch_size=64):  # one shard is a single 2,900-photo row group
            for row in batch.to_pylist():
                stats["photos"] += 1
                W, H, obj = row["width"], row["height"], row["objects"]
                boxes = [b for b, c in zip(obj["bbox"], obj["category"]) if c == person]
                if not boxes:
                    continue
                x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                bw, bh = x2 - x1, y2 - y1
                if bw * bh < MIN_AREA * W * H or bh < MIN_HEIGHT * H:
                    continue
                stats["with_prominent_person"] += 1
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                out.append({"bytes": png_bytes(img), "path": f"{row['image_id']}_full.png"})

                # square crop from the top of the person box: 0.45 = chest-up ... 1.0 = whole person
                side = max(bw * 1.15, rng.uniform(0.45, 1.0) * bh)
                side = min(side, W, H)
                if side >= MIN_CROP:
                    cx = (x1 + x2) / 2
                    left = float(np.clip(cx - side / 2, 0, W - side))
                    top = float(np.clip(y1 - 0.06 * side, 0, H - side))
                    crop = img.crop((round(left), round(top), round(left + side), round(top + side)))
                    out.append({"bytes": png_bytes(crop), "path": f"{row['image_id']}_person.png"})
                    stats["crops"] += 1
        print(f"{shard.name[:20]}: {stats}", flush=True)

    order = rng.permutation(len(out))  # mix full photos and crops across row groups
    table = pa.table({"image": [out[i] for i in order]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT, row_group_size=100)
    print(f"saved {OUT.name}: {len(out)} images ({stats['with_prominent_person']} full photos + {stats['crops']} "
          f"person crops) from {stats['photos']} COCO photos")
    print("COCO_PEOPLE_DONE")


if __name__ == "__main__":
    main()
