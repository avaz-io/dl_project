"""Report figures and summary tables from everything in runs/ and results/.

    python make_figures.py

Writes results/figures/*.png, results/summary.csv and results/summary.md.
Color = model (fixed order: baseline, ResNet-18, EfficientNet-B0); standard training is
drawn hatched/dashed, compression-aware training solid, so identity never relies on color alone.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"
FIG = RES / "figures"

MODELS = ["baseline", "resnet18", "efficientnet_b0"]
LABEL = {"baseline": "Baseline CNN", "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0"}
COLOR = {"baseline": "#2a78d6", "resnet18": "#eb6834", "efficientnet_b0": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
AUG_LABEL = {"standard": "standard training", "compress": "compression-aware"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False, "axes.axisbelow": True,
})


def load_runs():
    runs = []
    for res in sorted((ROOT / "runs").glob("*/results.json")):
        info = json.loads(res.read_text())
        a = info["args"]
        runs.append({"dir": res.parent, "name": res.parent.name, "model": a["model"],
                     "augment": "compress" if a.get("compress_aug") else "standard", "info": info,
                     "history": json.loads((res.parent / "history.json").read_text())})
    return runs


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def pick(runs, model, aug):
    c = [r for r in runs if r["model"] == model and r["augment"] == aug]
    return c[-1] if c else None  # newest if repeated


def bar_style(model, aug):
    if aug == "compress":
        return {"color": COLOR[model], "edgecolor": SURFACE, "linewidth": 1.5}
    return {"color": SURFACE, "edgecolor": COLOR[model], "hatch": "////", "linewidth": 1.2}


def fig_training(runs):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in MODELS:
        for aug in ["standard", "compress"]:
            r = pick(runs, m, aug)
            if not r:
                continue
            ep = [h["epoch"] for h in r["history"]]
            f1 = [100 * h["val_f1"] for h in r["history"]]
            ax.plot(ep, f1, color=COLOR[m], lw=2, ls="-" if aug == "compress" else "--", marker="o", ms=4,
                    label=f"{LABEL[m]}, {AUG_LABEL[aug]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation F1 (%)")
    ax.set_title("Training progress on CIFAKE (validation set, 10,000 images)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=170)
    plt.close(fig)


def fig_paired_bars(rows, key_col, keys, value_col, title, ylabel, fname, scale=100, ylim=(0, 105)):
    """One panel per model; per x key, a hatched bar (standard) and a solid bar (compression-aware)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    w = 0.38
    for ax, m in zip(axes, MODELS):
        x = np.arange(len(keys))
        for k, aug in enumerate(["standard", "compress"]):
            vals = []
            for key in keys:
                hit = [r for r in rows if r["model"] == m and r["augment"] == aug and r[key_col] == key]
                vals.append(scale * float(hit[-1][value_col]) if hit else np.nan)
            bars = ax.bar(x + (k - 0.5) * w, vals, w, label=AUG_LABEL[aug], **bar_style(m, aug))
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom",
                            fontsize=7, color=INK2)
        ax.set_xticks(x, keys, rotation=35, ha="right")
        ax.set_title(LABEL[m])
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontweight="bold")
    from matplotlib.patches import Patch  # one shared legend: the hatch, not the color, encodes training
    fig.legend(handles=[Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", label=AUG_LABEL["standard"]),
                        Patch(facecolor=INK2, edgecolor=SURFACE, label=AUG_LABEL["compress"])],
               loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / fname, dpi=170)
    plt.close(fig)


def fig_confusion(runs):
    show = [r for m in MODELS for aug in ["standard", "compress"] if (r := pick(runs, m, aug))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax in axes.flat:
        ax.set_visible(False)
    for i, r in enumerate(show):
        ax = axes[i % 2, MODELS.index(r["model"])]
        ax.set_visible(True)
        cm = np.array(r["info"]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() / 2)
        ax.grid(False)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.sum() / 4 else INK)
        ax.set_xticks([0, 1], ["pred. real", "pred. AI"])
        ax.set_yticks([0, 1], ["real", "AI"])
        ax.set_title(f"{LABEL[r['model']]}\n{AUG_LABEL[r['augment']]}", fontsize=10)
    fig.suptitle("Confusion matrices, CIFAKE test set (20,000 images)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=170)
    plt.close(fig)


V2_NAMES = {
    "ai_coco_sdxl": "SDXL scenes", "ai_ffhq_sdxl": "SDXL faces", "ai_genimage_ADM": "ADM", "ai_genimage_VQDM": "VQDM",
    "ai_genimage_Wukong": "Wukong", "ai_journeydb": "Midjourney (JourneyDB)", "ai_realvisxl": "RealVisXL",
    "real_celebahq": "CelebA-HQ faces", "real_coco": "MS-COCO scenes", "real_ffhq": "FFHQ faces",
    "real_caltech256": "Caltech-256 objects", "real_openimages": "Open Images photos",
    "ai_coco_flux": "FLUX scenes", "ai_ffhq_flux": "FLUX faces", "ai_genimage_BigGAN": "BigGAN",
    "ai_genimage_GLIDE": "GLIDE", "ai_genimage_Midjourney": "Midjourney (GenImage)", "real_imagenet": "ImageNet photos",
}


def fig_v2(rows):
    """v2 detector: share classified correctly per source, clean (solid) vs messenger re-share (hatched).
    Color = image type (blue real, orange AI); the two panels separate seen from held-out sources."""
    from matplotlib.patches import Patch
    groups = list(dict.fromkeys(r["group"] for r in rows))
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 5), sharex=True,
                             gridspec_kw={"width_ratios": [1] * len(groups)})
    for ax, g in zip(np.atleast_1d(axes), groups):
        srcs = sorted(dict.fromkeys(r["source"] for r in rows if r["group"] == g),
                      key=lambda x: (x.startswith("ai_"), x))
        y = np.arange(len(srcs))
        for k, cond in enumerate(["clean", "messenger"]):
            vals = [100 * float(next(r["correct_rate"] for r in rows
                                     if r["group"] == g and r["source"] == x and r["condition"] == cond)) for x in srcs]
            cols = [COLOR["resnet18"] if x.startswith("ai_") else COLOR["baseline"] for x in srcs]
            for yi, v, c in zip(y, vals, cols):
                style = {"color": c, "edgecolor": SURFACE, "linewidth": 1.5} if cond == "clean" else \
                        {"color": SURFACE, "edgecolor": c, "hatch": "////", "linewidth": 1.2}
                pos = yi + (0.19 if k == 0 else -0.19)  # clean bar above, messenger bar below
                ax.barh(pos, v, 0.38, **style)
                ax.text(v + 1, pos, f"{v:.0f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [V2_NAMES.get(x, x) for x in srcs])
        ax.set_xlim(0, 108)
        ax.set_xlabel("classified correctly (%)")
        ax.set_title(g[0].upper() + g[1:])
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(facecolor=COLOR["baseline"], label="real photos"),
                        Patch(facecolor=COLOR["resnet18"], label="AI-generated"),
                        Patch(facecolor=INK2, edgecolor=SURFACE, label="clean image"),
                        Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", label="after messenger-style re-share")],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, fontsize=9)
    fig.suptitle("v2 detector (EfficientNet-B0, 224×224, many generators)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "v2_results.png", dpi=170)
    plt.close(fig)



def summary(runs, robust, gen_rows):
    rows = []
    for m in MODELS:
        for aug in ["standard", "compress"]:
            r = pick(runs, m, aug)
            if not r:
                continue
            t = r["info"]["test"]
            rb = {x["condition"]: x for x in robust if x["run"] == r["name"]}
            g = [x for x in gen_rows if x["run"] == r["name"] and x["generator"] == "All generators"]
            rows.append({
                "model": LABEL[m], "training": AUG_LABEL[aug], "params_M": f"{r['info']['params_millions']:.2f}",
                "train_min": f"{r['info']['train_minutes']:.1f}", "accuracy": f"{100 * t['accuracy']:.2f}",
                "f1": f"{100 * t['f1']:.2f}", "roc_auc": f"{t['roc_auc']:.4f}",
                "false_pos_rate": f"{100 * t['false_positive_rate']:.2f}",
                "f1_jpeg50": f"{100 * float(rb['jpeg50']['f1']):.1f}" if "jpeg50" in rb else "",
                "f1_messenger": f"{100 * float(rb['messenger']['f1']):.1f}" if "messenger" in rb else "",
                "unseen_gen_balanced_acc": f"{100 * float(g[0]['balanced_accuracy']):.1f}" if g else "",
                "unseen_gen_detection": f"{100 * float(g[0]['detection_rate']):.1f}" if g else "",
            })
    with open(RES / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    head = ["Model", "Training", "Params (M)", "Train (min)", "Acc. %", "F1 %", "ROC-AUC", "FPR %",
            "F1 JPEG50 %", "F1 messenger %", "Unseen-gen bal. acc. %", "Unseen-gen detection %"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r.values()) + " |" for r in rows]
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    fig_training(runs)
    fig_confusion(runs)

    robust = read_csv(RES / "robustness.csv")
    if robust:
        conds = ["clean", "jpeg90", "jpeg50", "jpeg30", "scale0.75", "scale0.5", "scale0.4", "blur0.5",
                 "blur1.0", "messenger"]
        fig_paired_bars(robust, "condition", conds, "f1",
                        "Robustness on the CIFAKE test set: F1 under image degradations", "F1 (%)",
                        "robustness_f1.png")

    gen_rows = read_csv(RES / "genimage.csv")
    if gen_rows:
        gens = list(dict.fromkeys(r["generator"] for r in gen_rows))
        fig_paired_bars(gen_rows, "generator", gens, "detection_rate",
                        "Unseen generators (GenImage): share of AI images detected", "AI images detected (%)",
                        "genimage_detection.png")
        fig_paired_bars(gen_rows, "generator", gens, "balanced_accuracy",
                        "Unseen generators (GenImage): balanced accuracy (AI detection + real-photo accuracy)",
                        "balanced accuracy (%)", "genimage_balanced_accuracy.png")
    v2_rows = read_csv(RES / "v2_results.csv")
    if v2_rows:
        fig_v2(v2_rows)
    summary(runs, robust, gen_rows)
    print(f"figures saved in {FIG}")


if __name__ == "__main__":
    main()
