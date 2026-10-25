"""Train and evaluate the v2 detector: EfficientNet-B0 at 224x224 on many generators.

    python build_v2_dataset.py      # first
    python train_v2.py --epochs 12

Each batch is half real / half AI, with AI images drawn evenly across generators, and half of
the images get the compression-aware degradations from degrade.py. The model is selected on the
validation split of the *training* sources only; held-out generators are never used for selection.

Writes runs_v2/<name>/best.pt, history.json, results.json and results/v2_results.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from degrade import CONDITIONS, apply, random_degrade
from train import ROOT, to_tensor

V2 = ROOT / "data" / "v2"


def build_v2_model(pretrained=True):
    m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    return m


def gpu_augment(x, p_zoom=0.5):
    """Random horizontal flip, and for half of the batches a random zoom-crop (scale 0.75-1.0).

    The zoom resamples (slightly blurs) the batch. Applying it to every batch made the model treat
    untouched, sharper images - which is what it sees at test time - as AI, so half stay pristine."""
    n, _, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    if np.random.rand() >= p_zoom:
        return x.contiguous()
    s = np.random.uniform(0.75, 1.0)
    ch, cw = int(h * s), int(w * s)
    i, j = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
    return F.interpolate(x[:, :, i : i + ch, j : j + cw], size=(h, w), mode="bilinear", align_corners=False).contiguous()


@torch.no_grad()
def predict(model, images, idx, device, fn=None, bs=64):
    """Probabilities for images[idx]; fn is an optional PIL degradation applied first."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = np.asarray(images[np.sort(idx[i : i + bs])])
        if fn is not None:
            xb = apply(xb, fn)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(to_tensor(xb, device))
        out.append(torch.sigmoid(logits.float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


def report(model, images, meta, device, splits, fn=None):
    """Per-source share of correct predictions on the given splits (sorted index order)."""
    rows = []
    for src in sorted(set(meta["source"])):
        idx = np.sort(np.flatnonzero((meta["source"] == src) & np.isin(meta["split"], splits)))
        if len(idx) == 0:
            continue
        p = predict(model, images, idx, device, fn)
        is_ai = src.startswith("ai_")
        correct = (p >= 0.5) if is_ai else (p < 0.5)
        rows.append({"source": src, "label": "AI" if is_ai else "real", "n": len(idx),
                     "correct_rate": float(correct.mean()), "mean_score": float(p.mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=500, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--compress-p", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", default="", help="run folder to continue, or 'auto' for the newest unfinished run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    images = np.load(V2 / "images.npy", mmap_mode="r")
    meta = dict(np.load(V2 / "meta.npz"))
    train = meta["split"] == "train"
    real_idx = np.flatnonzero(train & (meta["label"] == 0))
    real_by_src = [np.flatnonzero(train & (meta["source"] == s)) for s in sorted(set(meta["source"][train]))
                   if s.startswith("real_")]
    ai_by_src = [np.flatnonzero(train & (meta["source"] == s)) for s in sorted(set(meta["source"][train]))
                 if s.startswith("ai_")]
    val_idx = np.flatnonzero(meta["split"] == "val")
    print(f"train: {len(real_idx)} real from {len(real_by_src)} sources, {sum(map(len, ai_by_src))} AI from "
          f"{len(ai_by_src)} generators | val {len(val_idx)}")

    model = build_v2_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    history, best, bad, t0, start = [], -1.0, 0, time.time(), 1
    run = None
    if args.resume == "auto":  # newest run that has a checkpoint but no final results
        unfinished = [d for d in sorted((ROOT / "runs_v2").glob("*")) if (d / "last.pt").exists()
                      and not (d / "results.json").exists()]
        run = unfinished[-1] if unfinished else None
    elif args.resume:
        run = Path(args.resume)
    if run is not None and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best, bad, start = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        t0 = time.time() - 60 * history[-1]["minutes"]
        rng = np.random.default_rng(args.seed + ck["epoch"])  # fresh batches for the remaining epochs
        print(f"resuming {run.name} from epoch {start}")
    else:
        run = ROOT / "runs_v2" / f"efficientnet_b0_224_{time.strftime('%Y%m%d_%H%M%S')}"
        run.mkdir(parents=True)
    half = args.bs // 2

    for epoch in range(start, args.epochs + 1):
        model.train()
        total = 0.0
        bar = tqdm(range(args.steps), desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for _ in bar:
            # even over sources on both sides, so no single source (e.g. faces) dominates a class
            ai = [rng.choice(ai_by_src[k]) for k in rng.integers(0, len(ai_by_src), half)]
            real = [rng.choice(real_by_src[k]) for k in rng.integers(0, len(real_by_src), half)]
            idx = np.sort(np.concatenate([real, ai]))
            xb = random_degrade(np.asarray(images[idx]), rng, args.compress_p)
            x = gpu_augment(to_tensor(xb, device))
            y = torch.from_numpy(meta["label"][idx]).float().to(device).unsqueeze(1)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")

        p = predict(model, images, val_idx, device)
        yv = meta["label"][np.sort(val_idx)]
        pred = p >= 0.5
        tpr, tnr = pred[yv == 1].mean(), (~pred[yv == 0]).mean()
        bal = (tpr + tnr) / 2
        history.append({"epoch": epoch, "train_loss": total / args.steps, "val_balanced_acc": float(bal),
                        "val_ai_detected": float(tpr), "val_real_correct": float(tnr),
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch:2d} | loss {total / args.steps:.4f} | val balanced acc {bal:.4f} "
              f"(AI detected {tpr:.3f}, real correct {tnr:.3f}) | {(time.time() - t0) / 60:.1f} min", flush=True)
        if bal > best:
            best, bad = bal, 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            bad += 1
        # checkpoint every epoch so an interrupted run can continue with --resume
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": float(best), "bad": bad, "history": history},
                   run / "last.pt")
        if bad:
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    rows = []
    for cond in ["clean", "jpeg50", "scale0.5", "blur1.0", "messenger"]:
        for group, splits in [("seen generators (test split)", ["test"]), ("held-out (unseen)", ["unseen"])]:
            for r in report(model, images, meta, device, splits, CONDITIONS[cond]):
                rows.append({"condition": cond, "group": group, **r})

    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "results.json").write_text(json.dumps(
        {"args": vars(args), "best_val_balanced_acc": best, "train_minutes": (time.time() - t0) / 60, "rows": rows},
        indent=2))
    out = ROOT / "results" / "v2_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nTEST RESULTS, clean images (share classified correctly)")
    for group in ["seen generators (test split)", "held-out (unseen)"]:
        print(f" {group}")
        for r in rows:
            if r["condition"] == "clean" and r["group"] == group:
                print(f"   {r['source']:26s} {r['label']:4s} n={r['n']:5d}  correct {100 * r['correct_rate']:5.1f}%")
    print(f"saved to {run} and {out}")
    print("V2_TRAINING_DONE")


if __name__ == "__main__":
    main()
