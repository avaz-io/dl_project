"""Check image files with the v2 (224x224, many-generator) detector.

    python predict.py photo.jpg another.png
    python predict.py --run runs_v2/efficientnet_b0_224_2026... photo.jpg

Prints the model score for "AI-generated" (0-100%). The score is evidence, not proof: it is
reliable mainly for the kinds of images and generators the model was trained on.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from build_v2_dataset import to_v2_format
from train import ROOT, to_tensor
from train_v2 import build_v2_model


def newest_run():
    runs = sorted(p.parent for p in (ROOT / "runs_v2").glob("*/results.json"))
    if not runs:
        raise SystemExit("no finished v2 run in runs_v2/ - train one with: python train_v2.py")
    return runs[-1]


def load_model(run, device):
    model = build_v2_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device, weights_only=True))
    return model.eval()


@torch.no_grad()
def score(model, img, device):
    """Probability (0-1) that a PIL image is AI-generated."""
    x = to_tensor(np.array(to_v2_format(img))[None], device)
    return float(torch.sigmoid(model(x).float())[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run = args.run or newest_run()
    model = load_model(run, device)
    print(f"model: {Path(run).name}")
    for path in args.images:
        img = Image.open(path)
        p = score(model, img, device)
        print(f"{'AI-generated' if p >= 0.5 else 'Real photo  '}  score {100 * p:5.1f}%  {img.size[0]}x{img.size[1]}  {path.name}")


if __name__ == "__main__":
    main()
