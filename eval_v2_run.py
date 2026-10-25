"""Evaluate one trained v2 run on the current v2 dataset (test split + held-out sources).

    python eval_v2_run.py runs_v2/efficientnet_b0_224_2026... results/v2_results_old_model.csv

Used to compare an older model with a newer one on exactly the same images. Note: for sources
the older model trained on, its "test split" numbers can be optimistic after a dataset rebuild
(the split is re-drawn); held-out and newly added sources are a fair comparison.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from degrade import CONDITIONS
from predict import load_model
from train_v2 import V2, report


def main(run, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(run, device)
    images = np.load(V2 / "images.npy", mmap_mode="r")
    meta = dict(np.load(V2 / "meta.npz"))
    rows = []
    for cond in ["clean", "messenger"]:
        for group, splits in [("seen generators (test split)", ["test"]), ("held-out (unseen)", ["unseen"])]:
            for r in report(model, images, meta, device, splits, CONDITIONS[cond]):
                rows.append({"condition": cond, "group": group, **r})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out} ({Path(run).name})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
