"""Train and evaluate real-vs-AI image detectors on CIFAKE.

Examples:
    python train.py --model baseline --epochs 15
    python train.py --model resnet18 --epochs 10 --lr 1e-3
    python train.py --model resnet18 --epochs 10 --compress-aug   # O3: compression-aware training

Each run writes runs/<model>[_compress]_<timestamp>/ with best.pt, history.json and results.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

from degrade import random_degrade

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cifake"
CACHE = ROOT / "data" / "cifake_cache.npz"
CLASSES = ["REAL", "FAKE"]  # label 0 = real, 1 = AI-generated
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet statistics
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def load_split(split):
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        files = sorted((DATA / split / cls).glob("*"))
        for f in tqdm(files, desc=f"reading {split}/{cls}", leave=False):
            with Image.open(f) as im:
                xs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        ys += [label] * len(files)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_cifake():
    """Return uint8 arrays (N, 32, 32, 3) and labels; decoded once, then cached."""
    if not CACHE.exists():
        print("First run: decoding 120,000 images into a cache (takes a minute)...")
        xtr, ytr = load_split("train")
        xte, yte = load_split("test")
        np.savez(CACHE, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    d = np.load(CACHE)
    return d["xtr"], d["ytr"], d["xte"], d["yte"]


def stratified_val_split(y, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    val = np.concatenate([rng.permutation(np.where(y == c)[0])[: int(frac * (y == c).sum())] for c in (0, 1)])
    train = np.setdiff1d(np.arange(len(y)), val)
    return train, val


def to_tensor(x_uint8, device):
    """uint8 NHWC numpy -> normalized float NCHW tensor on device."""
    # .contiguous(): permute only re-labels the axes, leaving NHWC memory; feeding that to an NCHW
    # model makes cuDNN ~13x slower with mixed precision at 224x224.
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).to(device).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x):
    """Standard augmentation on a GPU batch: random horizontal flip + random crop (pad 2)."""
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    i, j = np.random.randint(0, 5, size=2)
    return x[:, :, i : i + 32, j : j + 32]


# ---------------------------------------------------------------- models
class CifakeCNN(nn.Module):
    """Baseline from Bird & Lotfi (CIFAKE paper): 2 conv layers x 32 filters + 2 dense layers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x))


class Upsampled(nn.Module):
    """Resize 32x32 inputs before an ImageNet backbone that expects larger images."""

    def __init__(self, net, size):
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x):
        return self.net(F.interpolate(x, size=self.size, mode="bilinear", align_corners=False))


def build_model(name):
    if name == "baseline":
        return CifakeCNN()
    if name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        # Adapt to 32x32 inputs: 3x3 stride-1 first conv, no max-pool (keeps spatial detail).
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    if name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return Upsampled(m, 128)
    raise ValueError(f"unknown model {name}")


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, x_uint8, device, bs=1024):
    model.eval()
    probs = []
    for i in range(0, len(x_uint8), bs):
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(to_tensor(x_uint8[i : i + bs], device))
        probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu())
    return torch.cat(probs).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "false_positive_rate": fp / (fp + tn),  # real photos wrongly flagged as AI
        "false_negative_rate": fn / (fn + tp),  # AI images that slipped through
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline", choices=["baseline", "resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--compress-aug", action="store_true",
                    help="O3: randomly JPEG-compress / rescale / blur training images")
    ap.add_argument("--compress-p", type=float, default=0.5, help="fraction of images degraded per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (quick debugging)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    xtr_all, ytr_all, xte, yte = load_cifake()
    tr_idx, va_idx = stratified_val_split(ytr_all, 0.1, args.seed)
    if args.limit:
        tr_idx = np.random.permutation(tr_idx)[: args.limit]
    xtr, ytr, xva, yva = xtr_all[tr_idx], ytr_all[tr_idx], xtr_all[va_idx], ytr_all[va_idx]
    print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)}")

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} ({n_params / 1e6:.2f}M parameters)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(ytr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    tag = "_compress" if args.compress_aug else ""
    run = ROOT / "runs" / f"{args.model}{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    history, best_f1, bad_epochs, t0 = [], -1.0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(ytr))
        total = 0.0
        bar = tqdm(range(len(ytr) // args.bs), desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for b in bar:
            idx = perm[b * args.bs : (b + 1) * args.bs]
            xb = xtr[idx]
            if args.compress_aug:
                xb = random_degrade(xb, rng, args.compress_p)
            x = to_tensor(xb, device)
            if not args.no_augment:
                x = augment(x)
            y = torch.from_numpy(ytr[idx]).float().to(device).unsqueeze(1)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")

        val = metrics(yva, predict(model, xva, device))
        row = {"epoch": epoch, "train_loss": total / (b + 1), "val_acc": val["accuracy"], "val_f1": val["f1"],
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(f"epoch {epoch:2d} | loss {row['train_loss']:.4f} | val acc {val['accuracy']:.4f} "
              f"| val F1 {val['f1']:.4f} | {row['minutes']:.1f} min")

        if val["f1"] > best_f1:
            best_f1, bad_epochs = val["f1"], 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no val F1 improvement for {args.patience} epochs)")
                break

    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    test = metrics(yte, predict(model, xte, device))
    results = {"args": vars(args), "params_millions": n_params / 1e6,
               "train_minutes": (time.time() - t0) / 60, "best_val_f1": best_f1, "test": test}
    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "results.json").write_text(json.dumps(results, indent=2, default=float))

    print("\nTEST RESULTS (20,000 images)")
    for k in ["accuracy", "f1", "roc_auc", "false_positive_rate", "false_negative_rate"]:
        print(f"  {k:20s} {test[k]:.4f}")
    print(f"  confusion matrix [[TN, FP], [FN, TP]] = {test['confusion_matrix']}")
    print(f"saved to {run}")


if __name__ == "__main__":
    main()
