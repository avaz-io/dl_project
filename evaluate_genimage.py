"""Objective O2: how well do CIFAKE-trained detectors generalize to unseen generators?

    python evaluate_genimage.py            # every finished run in runs/
    python build_genimage_test.py          # (run first) builds data/genimage_test.npz

All generators share one set of real ImageNet photos, so for each generator we report:
  detection rate  - share of that generator's AI images flagged as AI (true positive rate)
  balanced acc.   - mean of detection rate and the real-photo accuracy (true negative rate)
Writes results/genimage.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from robustness import load_run
from train import ROOT, predict

REAL = "Real (ImageNet)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "genimage.csv")
    args = ap.parse_args()

    d = np.load(ROOT / "data" / "genimage_test.npz")
    x, y, gen = d["x"], d["y"], d["gen"]
    gens = [g for g in dict.fromkeys(gen) if g != REAL]
    print(f"test set: {int((y == 0).sum())} real ImageNet photos, AI images per generator: "
          + ", ".join(f"{g} {int((gen == g).sum())}" for g in gens))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dirs = args.runs or sorted(p.parent for p in (ROOT / "runs").glob("*/results.json"))
    rows = []
    for run in run_dirs:
        m = load_run(run, device)
        p = predict(m["net"], x, device)
        pred = (p >= 0.5).astype(int)
        tnr = (pred[y == 0] == 0).mean()  # real photos correctly called real
        for g in gens + ["All generators"]:
            sel = (gen == g) if g != "All generators" else (y == 1)
            tpr = (pred[sel] == 1).mean()
            both = sel | (y == 0)
            rows.append({"run": m["run"], "model": m["model"], "augment": m["augment"], "generator": g,
                         "n_ai": int(sel.sum()), "detection_rate": tpr, "real_accuracy": tnr,
                         "balanced_accuracy": (tpr + tnr) / 2, "roc_auc": roc_auc_score(y[both], p[both])})
        del m

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = gens + ["All generators"]
    runs = list(dict.fromkeys(r["run"] for r in rows))
    width = max(len(r) for r in runs)
    for metric, title in [("detection_rate", "AI images detected (%)"),
                          ("balanced_accuracy", "Balanced accuracy (%)"), ("roc_auc", "ROC-AUC")]:
        print(f"\n{title}")
        print(" " * width + "".join(f"{c[:14]:>15}" for c in cols) + f"{'real acc.':>12}")
        for run in runs:
            cells = {r["generator"]: r for r in rows if r["run"] == run}
            scale = 1 if metric == "roc_auc" else 100
            print(f"{run:<{width}}" + "".join(f"{scale * cells[c][metric]:>15.{3 if scale == 1 else 1}f}" for c in cols)
                  + f"{100 * cells[cols[0]]['real_accuracy']:>12.1f}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
