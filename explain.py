"""Objective O4: what do the detectors look at?

    python explain.py spectrum            # frequency analysis of real vs AI images (CPU, no model needed)
    python explain.py gradcam             # Grad-CAM for 100 test images, every finished run
    python explain.py gradcam runs/resnet18_2026...

Outputs go to results/: spectrum.png, spectrum_profile.csv, gradcam_<run>.png, gradcam_<run>_examples.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from degrade import CONDITIONS, apply
from train import ROOT, build_model, load_cifake, to_tensor

OUT = ROOT / "results"


# ---------------------------------------------------------------- frequency analysis
def power_spectrum(x_uint8):
    """Mean 2D power spectrum (centered) of grayscale images, each image mean-subtracted."""
    g = x_uint8.astype(np.float64).mean(-1)
    g -= g.mean(axis=(1, 2), keepdims=True)
    return (np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2).mean(0)


def radial_profile(spec):
    """Average power at each integer distance from the center (0 = lowest frequency)."""
    h, w = spec.shape
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    prof = np.bincount(r.ravel(), spec.ravel()) / np.bincount(r.ravel())
    return prof[: h // 2 + 1]  # up to the Nyquist frequency


def spectrum():
    _, _, xte, yte = load_cifake()
    real, fake = xte[yte == 0], xte[yte == 1]
    curves = {
        "real": real,
        "AI-generated": fake,
        "AI-generated + blur sigma=1": apply(fake, CONDITIONS["blur1.0"]),
        "AI-generated + downscale 0.75": apply(fake, CONDITIONS["scale0.75"]),
        "AI-generated + messenger": apply(fake, CONDITIONS["messenger"]),
    }
    specs = {k: power_spectrum(v) for k, v in curves.items()}
    profiles = {k: radial_profile(s) for k, s in specs.items()}

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    lr, lf = np.log10(specs["real"]), np.log10(specs["AI-generated"])
    c = lr.shape[0] // 2
    lr[c, c] = lf[c, c] = np.nan  # zero frequency is ~0 after mean subtraction; hide it
    vmin, vmax = np.nanmin([lr, lf]), np.nanmax([lr, lf])
    for a, img, title in [(ax[0], lr, "Real: mean log power spectrum"),
                          (ax[1], lf, "AI-generated: mean log power spectrum")]:
        im = a.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        a.set_title(title)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    diff = lf - lr
    lim = np.nanmax(np.abs(diff))
    im = ax[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title("Difference (AI − real), log10")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    styles = {"real": ("k", "-"), "AI-generated": ("tab:red", "-")}
    freq = np.arange(len(profiles["real"])) / (2 * (len(profiles["real"]) - 1))
    for k, p in profiles.items():
        color, ls = styles.get(k, (None, "--"))
        ax[3].plot(freq[1:], p[1:] / profiles["real"][1:], color=color, ls=ls, label=k)  # skip zero frequency
    ax[3].set_xlabel("spatial frequency (cycles/pixel)")
    ax[3].set_ylabel("power relative to real images")
    ax[3].set_title("Radially averaged power spectrum")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "spectrum.png", dpi=150)

    with open(OUT / "spectrum_profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frequency_cycles_per_pixel"] + list(profiles))
        n = len(profiles["real"])
        for i in range(n):
            w.writerow([i / (2 * (n - 1))] + [profiles[k][i] for k in profiles])

    hi = slice(len(profiles["real"]) // 2, None)  # upper half of the frequency range
    print("High-frequency power relative to real images (upper half of spectrum):")
    for k, p in profiles.items():
        print(f"  {k:32s} {p[hi].mean() / profiles['real'][hi].mean():6.2f}x")
    print(f"saved {OUT / 'spectrum.png'} and {OUT / 'spectrum_profile.csv'}")


# ---------------------------------------------------------------- Grad-CAM
def target_layer(model, name):
    if name == "baseline":
        return model.features[4]  # ReLU after the 2nd conv (13x13 map)
    if name == "resnet18":
        return model.layer4  # 4x4 map for 32x32 input
    if name == "efficientnet_b0":
        return model.net.features[-1]  # 4x4 map for 128x128 input
    raise ValueError(name)


def gradcam(model, layer, x):
    """Grad-CAM for the predicted class of each image. Returns (cams[N,32,32] in 0..1, prob_fake[N])."""
    store = {}

    def fwd(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        logits = model(x).squeeze(1)
        sign = torch.where(logits >= 0, 1.0, -1.0)  # explain "fake" if predicted fake, else "real"
        (sign * logits).sum().backward()
    finally:
        h.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8).view(-1, 1, 1)
    return cam.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()


def gradcam_runs(run_dirs, n, cpu):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    _, _, xte, yte = load_cifake()
    rng = np.random.default_rng(0)  # same 100 images for every model
    idx = np.concatenate([rng.choice(np.flatnonzero(yte == c), n // 2, replace=False) for c in (0, 1)])
    x_img, y = xte[idx], yte[idx]
    OUT.mkdir(exist_ok=True)

    for run in run_dirs:
        args = json.loads((run / "results.json").read_text())["args"]
        model = build_model(args["model"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
        model.eval()
        cams, prob = gradcam(model, target_layer(model, args["model"]), to_tensor(x_img, device))
        pred = (prob >= 0.5).astype(int)

        # Full grid: all n images with heatmap overlay; green title = correct, red = wrong
        cols = 10
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.45))
        for i, a in enumerate(axes.flat):
            a.axis("off")
            if i >= n:
                continue
            a.imshow(x_img[i])
            a.imshow(cams[i], cmap="jet", alpha=0.45)
            truth = "AI" if y[i] else "Real"
            a.set_title(f"{truth} p={prob[i]:.2f}", fontsize=6.5, color="green" if pred[i] == y[i] else "red")
        fig.suptitle(f"Grad-CAM, {run.name}: top half real, bottom half AI-generated "
                     f"(p = probability AI; red = wrong). Accuracy on these {n}: {(pred == y).mean():.0%}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"gradcam_{run.name}.png", dpi=130)
        plt.close(fig)

        # Small figure for the report: 6 real + 6 AI, image row and overlay row, plus mean heatmap per class
        pick = list(range(6)) + list(range(n // 2, n // 2 + 6))
        fig, axes = plt.subplots(2, 14, figsize=(18, 3.2), gridspec_kw={"width_ratios": [1] * 12 + [1.15, 1.15]})
        for j, i in enumerate(pick):
            axes[0, j].imshow(x_img[i])
            axes[0, j].set_title(("AI" if y[i] else "Real") + f"\np={prob[i]:.2f}", fontsize=8,
                                 color="green" if pred[i] == y[i] else "red")
            axes[1, j].imshow(x_img[i])
            axes[1, j].imshow(cams[i], cmap="jet", alpha=0.5)
        for c, j in [(0, 12), (1, 13)]:
            axes[0, j].imshow(cams[y == c].mean(0), cmap="jet", vmin=0, vmax=1)
            axes[0, j].set_title(f"mean CAM\n{'AI' if c else 'Real'}", fontsize=8)
            axes[1, j].set_visible(False)
        for a in axes.flat:
            a.axis("off")
        fig.suptitle(f"Grad-CAM examples, {run.name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"gradcam_{run.name}_examples.png", dpi=150)
        plt.close(fig)
        print(f"{run.name}: accuracy on the {n} images {(pred == y).mean():.0%}, saved gradcam_{run.name}*.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spectrum")
    g = sub.add_parser("gradcam")
    g.add_argument("runs", nargs="*", type=Path)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cmd == "spectrum":
        spectrum()
    else:
        runs = args.runs or sorted(p.parent for p in (ROOT / "runs").glob("*/results.json"))
        gradcam_runs(runs, args.n, args.cpu)


if __name__ == "__main__":
    main()
