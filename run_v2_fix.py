"""One-shot pipeline for a v2 data fix. Run detached; it notifies on Telegram when done.

    python run_v2_fix.py

Steps: wait for download_v2.py to finish -> rebuild the 224x224 dataset -> evaluate the previous
model on the new dataset (same images, fair "before") -> retrain v2 -> compare -> notify.py.

This round adds real MS-COCO photos in which a person is prominent, plus chest-up / waist-up /
full-body crops of them (build_coco_people.py), because v2 called real photos AI as soon as a
person's torso was in frame.
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable
DOWNLOAD_LOG = "download_coco_log.txt"
NEW_SOURCES = ["real_coco_people"]
PREVIOUS_RUN = ROOT / "runs_v2" / "efficientnet_b0_224_20261013_171633"  # v2c
BEFORE_CSV = ROOT / "results" / "v2_results_v2c_on_new_data.csv"
# Optional extra test images, scored before and after: V2_TEST_IMAGES="label=path;label=path"
PHOTOS = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]))
          for item in os.environ.get("V2_TEST_IMAGES", "").split(";") if "=" in item]


def run(args, log):
    with open(ROOT / log, "w", encoding="utf-8") as f:
        return subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def clean_rates(path):
    rows = csv.DictReader(open(path, encoding="utf-8"))
    return {r["source"]: 100 * float(r["correct_rate"]) for r in rows if r["condition"] == "clean"}


def notify(text, files=()):
    args = ["notify.py", text]
    for f in files:
        args += ["--file", str(f)]
    subprocess.run([PY, *args], cwd=ROOT)


def score_line(run_dir, path):
    args = ["predict.py", str(path)] + (["--run", str(run_dir)] if run_dir else [])
    out = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True).stdout
    line = [ln for ln in out.splitlines() if path.name in ln]
    m = re.match(r"\s*(.*?score\s+[\d.]+%)", line[0]) if line else None
    return " ".join(m.group(1).split()) if m else "n/a"  # e.g. "AI-generated score 99.9%"


def main():
    log = ROOT / DOWNLOAD_LOG
    while not (log.exists() and "DOWNLOADS_DONE" in log.read_text(encoding="utf-8", errors="replace")):
        time.sleep(20)
    raw = ROOT / "data" / "v2" / "raw"
    if "FAIL" in log.read_text(encoding="utf-8", errors="replace") or run(["build_coco_people.py"], "build_coco_people_log.txt") != 0:
        notify("v2 chest-up fix: the COCO download or the photo selection failed (see download_coco_log.txt, "
               "build_coco_people_log.txt). Nothing was retrained; the current model is unchanged.")
        return
    missing = [n for n in NEW_SOURCES if not (raw / f"{n}.parquet").exists()]
    if missing:
        notify(f"v2 chest-up fix: {', '.join(missing)} was not produced. Nothing was retrained; the current model is unchanged.")
        return
    if run(["build_v2_dataset.py"], "build_v2_log.txt") != 0:
        notify("v2 chest-up fix: rebuilding the dataset failed (see build_v2_log.txt). The current model is unchanged.")
        return
    photos_before = {label: score_line(PREVIOUS_RUN, p) for label, p in PHOTOS if p.exists()}
    run(["eval_v2_run.py", str(PREVIOUS_RUN), str(BEFORE_CSV)], "eval_v2_before_log.txt")

    code = run(["train_v2.py", "--epochs", "12"], "train_v2_fix_log.txt")
    if code != 0:  # one automatic retry from the last checkpoint
        code = run(["train_v2.py", "--epochs", "12", "--resume", "auto"], "train_v2_fix_resume_log.txt")
    if code != 0:
        notify("v2 chest-up fix: training failed twice (see train_v2_fix_log.txt). The previous model is still in place.")
        return

    before = clean_rates(BEFORE_CSV) if BEFORE_CSV.exists() else {}
    after = clean_rates(ROOT / "results" / "v2_results.csv")
    subprocess.run([PY, "make_figures.py"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def pair(src):
        return f"{before[src]:.0f}% -> {after[src]:.0f}%" if src in before else f"{after[src]:.0f}%"

    names = [("real_coco_people", "real people, chest-up to full body, COCO (new)"),
             ("real_camera", "REAL full-size camera photos (never trained on)"),
             ("real_imagenet", "REAL ImageNet photos (never trained on)"),
             ("ai_genimage_Midjourney", "AI Midjourney (never seen)"), ("ai_ffhq_flux", "AI FLUX faces (never seen)"),
             ("ai_coco_flux", "AI FLUX scenes (never seen)"), ("ai_genimage_GLIDE", "AI GLIDE (never seen)"),
             ("ai_genimage_BigGAN", "AI BigGAN (never seen)"), ("ai_ffhq_sdxl", "AI SDXL portraits (trained on)"),
             ("ai_realvisxl", "AI RealVisXL (trained on)")]
    lines = [f"- {label}: {pair(src)}" for src, label in names if src in after]
    photo_lines = [f"- {label}: {photos_before.get(label, 'n/a')} -> {score_line(None, p)}"
                   for label, p in PHOTOS if p.exists()]
    notify("v2 chest-up fix: retraining finished.\n\nShare classified correctly, before -> after adding real COCO "
           "photos of people from the chest up / full body:\n" + "\n".join(lines) + "\n\nYour two test images "
           "(model verdict and AI score, before -> after):\n" + "\n".join(photo_lines) +
           "\n\nThe app uses the new model. It also now reads the file's own metadata first: with camera data "
           "present and a model verdict of AI, it shows 'Inconclusive' instead of 'AI-generated'.",
           [ROOT / "results" / "figures" / "v2_results.png"])
    print("V2_FIX_DONE")


if __name__ == "__main__":
    main()
