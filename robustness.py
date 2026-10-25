"""Objective O3: evaluate trained models on the CIFAKE test set under image degradations.

Examples:
    python robustness.py                       # every finished run in runs/
    python robustness.py runs/resnet18_2026...  # specific runs
    python robustness.py --limit 500 --cpu     # quick smoke test

Writes results/robustness.csv (one row per run x condition) and prints F1 and the
F1 drop versus clean images, in percentage points.
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from degrade import CONDITIONS, apply
from train import ROOT, build_model, load_cifake, metrics, predict


def load_run(run_dir, device):
    info = json.loads((run_dir / "results.json").read_text())
    model = build_model(info["args"]["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()
    augment = "compress" if info["args"].get("compress_aug") else "standard"
    return {"run": run_dir.name, "model": info["args"]["model"], "augment": augment, "net": model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path, help="run folders (default: all finished runs)")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=0, help="use only N test images per class (quick test)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "robustness.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dirs = args.runs or sorted(p.parent for p in (ROOT / "runs").glob("*/results.json"))
    if not run_dirs:
        raise SystemExit("no finished runs found in runs/")
    models = [load_run(r, device) for r in run_dirs]
    print(f"device: {device} | models: {', '.join(m['run'] for m in models)}")

    _, _, xte, yte = load_cifake()
    if args.limit:  # balanced subset: first N real + first N fake (test set is ordered by class)
        n_real = int((yte == 0).sum())
        keep = list(range(args.limit)) + list(range(n_real, n_real + args.limit))
        xte, yte = xte[keep], yte[keep]

    rows, clean_f1 = [], {}
    for cond in ["clean"] + [c for c in args.conditions if c != "clean"]:
        x = apply(xte, CONDITIONS[cond])  # degrade once, evaluate every model on the same images
        for m in models:
            r = metrics(yte, predict(m["net"], x, device))
            if cond == "clean":
                clean_f1[m["run"]] = r["f1"]
            rows.append({"run": m["run"], "model": m["model"], "augment": m["augment"], "condition": cond,
                         "accuracy": r["accuracy"], "f1": r["f1"], "roc_auc": r["roc_auc"],
                         "false_positive_rate": r["false_positive_rate"],
                         "f1_drop_pts": 100 * (clean_f1[m["run"]] - r["f1"])})
        print(f"  done: {cond}")

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Table: F1 (%) per condition, with drop vs clean in brackets
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    name_w = max(len(m["run"]) for m in models)
    print("\nF1 % (drop vs clean, points)")
    print(" " * name_w + "".join(f"{c:>16}" for c in conds))
    for m in models:
        cells = {r["condition"]: r for r in rows if r["run"] == m["run"]}
        line = "".join(f"{100 * cells[c]['f1']:>8.2f} ({cells[c]['f1_drop_pts']:>4.1f})" for c in conds)
        print(f"{m['run']:<{name_w}}{line}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
