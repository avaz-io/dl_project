"""Streamlit app: check an image for AI generation and browse the project results.

    .venv\\Scripts\\streamlit run app.py      (opens http://localhost:8501)
"""
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

import watermark_client
from build_genimage_test import to_cifake_format
from build_v2_dataset import to_v2_format
from explain import gradcam, target_layer
from portrait import ZONE, portrait_zone
from provenance import combine, inspect
from robustness import load_run
from train import ROOT, to_tensor
from train_v2 import build_v2_model

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
LABEL = {"baseline": "Baseline CNN", "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0"}
AUG = {"standard": "standard training", "compress": "compression-aware"}

st.set_page_config(page_title="Real or AI?", page_icon="🔍", layout="wide")


@st.cache_data
def list_runs():
    runs = []
    for res in sorted((ROOT / "runs").glob("*/results.json")):
        info = json.loads(res.read_text())
        a = info["args"]
        aug = "compress" if a.get("compress_aug") else "standard"
        runs.append({"dir": str(res.parent), "label": f"{LABEL[a['model']]} ({AUG[aug]})",
                     "model": a["model"], "aug": aug, "acc": info["test"]["accuracy"]})
    for r in runs:
        r["kind"] = "v1"
    # best first: EfficientNet-B0 compression-aware is the recommended 32x32 model
    order = {("efficientnet_b0", "compress"): 0, ("resnet18", "compress"): 1, ("efficientnet_b0", "standard"): 2}
    runs = sorted(runs, key=lambda r: order.get((r["model"], r["aug"]), 9))
    # v2: 224x224 model trained on many generators - listed first, meant for large real-world images
    v2 = [{"dir": str(res.parent), "label": "v2 · EfficientNet-B0 224×224 (many generators, for large images)",
           "model": "efficientnet_b0_v2", "aug": "compress", "kind": "v2"}
          for res in sorted((ROOT / "runs_v2").glob("*/results.json"))][-1:]  # newest v2 run
    return v2 + runs


@st.cache_resource
def get_model(run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(run_dir).parent.name == "runs_v2":
        net = build_v2_model(pretrained=False).to(device)
        net.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True))
        net.eval()
        return {"net": net}, device
    m = load_run(Path(run_dir), device)
    return m, device


def analyse(img, run):
    """Return (image as the model sees it, probability AI, Grad-CAM map)."""
    m, device = get_model(run["dir"])
    if run["kind"] == "v2":
        x = np.array(to_v2_format(img))[None]  # 224x224, same preprocessing as v2 training
        layer = m["net"].features[-1]
    else:
        x = np.array(to_cifake_format(img))[None]  # 32x32, same preprocessing as the test sets
        layer = target_layer(m["net"], run["model"])
    cams, prob = gradcam(m["net"], layer, to_tensor(x, device))
    return x[0], float(prob[0]), cams[0]


@st.cache_data
def v2_rates():
    """Clean-image results of the v2 model per source (results/v2_results.csv)."""
    path = RES / "v2_results.csv"
    if not path.exists():
        return []
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["condition"] == "clean"]


@st.cache_data
def unseen_detection_rate(run):
    """Share of unseen-generator AI images this run detected (results/genimage.csv), or None."""
    path = RES / "genimage.csv"
    if not path.exists():
        return None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["run"] == Path(run["dir"]).name and r["generator"] == "All generators":
            return 100 * float(r["detection_rate"])
    return None


@st.fragment(run_every=4)
def watermark_poll(h):
    """While the watermark check is pending, ask the second PC every 4 s; rerun the app when answered."""
    job = st.session_state["wm_jobs"].get(h)
    if not job:
        return
    try:
        status, watermark = watermark_client.status(job)
    except Exception as e:
        st.caption(f"Watermark check not reachable right now ({type(e).__name__}); still waiting.")
        return
    if status == "answered":
        st.session_state["wm_results"][h] = {"watermark": watermark}
        st.session_state["wm_jobs"].pop(h, None)
        st.rerun()
    st.info("⏳ Checking for an AI watermark…")


def heatmap_on_image(img, cam, max_side=900):
    """The uploaded image with the Grad-CAM heatmap drawn on the centre square the model analysed;
    the rest of the image is dimmed and the analysed square outlined."""
    import matplotlib.cm as cm
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2  # same centre square as the model's preprocessing
    base = np.asarray(im).astype(float) / 255
    out = base * 0.45
    heat = cm.jet(np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR))
                  / 255)[..., :3]
    out[top:top + side, left:left + side] = 0.55 * base[top:top + side, left:left + side] + 0.45 * heat
    b = max(2, side // 200)  # outline
    out[top:top + b, left:left + side] = out[top + side - b:top + side, left:left + side] = 1
    out[top:top + side, left:left + b] = out[top:top + side, left + side - b:left + side] = 1
    return out.clip(0, 1)


def example_images():
    ex = []
    for d in sorted((ROOT / "data" / "genimage" / "test").glob("*")):
        files = sorted(d.glob("*.jpg"))
        if files:
            ex.append((d.name, files))
    cifake = ROOT / "data" / "cifake" / "test"
    for cls in ["REAL", "FAKE"]:
        files = sorted((cifake / cls).glob("*"))[:200]
        if files:
            ex.append((f"CIFAKE {cls.lower()} (32×32)", files))
    return ex


# ---------------------------------------------------------------- UI
st.title("🔍 Real or AI? — AI-generated image detector")
st.caption("Deep Learning course project · 32×32 detectors trained on CIFAKE, plus a 224×224 v2 detector trained on "
           "many generators for large real-world images")

tab_check, tab_results, tab_about = st.tabs(["Check an image", "Results", "About"])

with tab_check:
    runs = list_runs()
    if not runs:
        st.error("No trained models found in runs/. Train one with `python train.py`.")
        st.stop()
    left, right = st.columns([1, 2])
    with left:
        run = runs[[r["label"] for r in runs].index(
            st.selectbox("Model", [r["label"] for r in runs], help="Compression-aware models stay accurate on "
                         "resized, blurred or messenger-compressed images."))]
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        examples = example_images()
        if examples:
            src = st.selectbox("…or try an example from", ["—"] + [n for n, _ in examples])
            if src != "—" and st.button("🎲 Random example"):
                st.session_state["example"] = str(random.choice(dict(examples)[src]))
        if run["kind"] == "v2":
            unseen = {r["source"]: 100 * float(r["correct_rate"]) for r in v2_rates() if "unseen" in r["group"]}
            ai = [v for k, v in unseen.items() if k.startswith("ai_")]
            note = ""
            if ai and "real_imagenet" in unseen:
                real = unseen["real_imagenet"]
                note = (f" Measured on data it never saw: it catches {min(ai):.0f}–{max(ai):.0f}% of AI images from "
                        f"unseen generators and gets **{real:.0f}%** of unfamiliar real photos (ImageNet) right"
                        + (" — so an \"AI\" verdict on an object or animal photo may be a false alarm." if real < 90 else
                           ". Images from generators it never saw are its weakest point."))
            st.info("v2 looks at a 224×224 center crop and was trained on real photos (faces, everyday scenes, objects) and "
                    "AI images from several generators (SDXL, RealVisXL, Midjourney, ADM, VQDM, Wukong)." + note,
                    icon="ℹ️")
        else:
            st.info("This model was trained on 32×32 CIFAKE images, so every picture is center-cropped and shrunk to "
                    "32×32 first. For large real-world photos use the v2 model.", icon="ℹ️")

    img, name, raw = None, None, None
    if up is not None:
        raw, name = up.getvalue(), up.name
        st.session_state.pop("example", None)
    elif "example" in st.session_state:
        raw, name = Path(st.session_state["example"]).read_bytes(), Path(st.session_state["example"]).parent.name
    if raw is not None:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))  # phone photos are often stored rotated

    with right:
        if img is None:
            st.write("⬅️ Upload an image or pick an example to start.")
        else:
            h = hashlib.sha1(raw).hexdigest()
            jobs = st.session_state.setdefault("wm_jobs", {})
            results = st.session_state.setdefault("wm_results", {})
            # 1. Every image goes to the watermark check on the second PC automatically (once per image)
            if watermark_client.configured() and h not in jobs and h not in results:
                try:
                    jobs[h] = watermark_client.send(raw, name or "image")
                except Exception as e:
                    results[h] = {"watermark": None, "error": type(e).__name__}
            wm = results.get(h)
            watermark = wm["watermark"] if wm else None

            # 2. File evidence and the model
            x, p_ai, cam = analyse(img, run)  # x: the image as the model sees it (32x32, or 224x224 for v2)
            evidence = inspect(raw)
            face = portrait_zone(img)  # None if no face was found
            kind, headline, why = combine(evidence, p_ai, watermark, face)

            st.markdown("##### 1 · Watermark check")
            if h in jobs:
                watermark_poll(h)
            elif wm and wm.get("error"):
                st.caption(f"Watermark check not reachable ({wm['error']}); the verdict uses the model only.")
            elif wm:
                {True: lambda: st.error("**AI watermark found** in this image.", icon="🏷️"),
                 False: lambda: st.info("No AI watermark found. This does not mean the image is real: most AI tools "
                                        "add no watermark.", icon="ℹ️"),
                 None: lambda: st.caption("The watermark check could not be done for this image.")}[wm["watermark"]]()
            else:
                st.caption("Watermark check not set up (see README); the verdict uses the file and the model only.")
            if watermark_client.configured():
                st.caption("Images opened here are also sent to an external watermark check.")

            st.markdown("##### 2 · Verdict")
            icon = {"ai_declared": "🏷️", "real_verified": "✅", "ai": "🤖", "real": "📷", "inconclusive": "❓"}[kind]
            st.subheader(f"{icon} {headline}")
            if h in jobs:
                st.caption("Preliminary: the watermark check is still running, and the verdict updates when it answers.")
            # A sigmoid score is not a calibrated confidence, especially far from the training data,
            # so show it as a model score rather than "% confident".
            st.progress(p_ai, text=f"model score (AI-generated): {100 * p_ai:.1f}%")
            (st.warning if kind == "inconclusive" else st.caption)(why)
            if 0.35 < p_ai < 0.65:
                st.warning("The model is unsure about this image. Treat its score as inconclusive.")
            if run["kind"] == "v2":
                if min(img.size) < 128:
                    st.warning(f"This image is only {img.size[0]}×{img.size[1]} px. v2 was trained on larger images; "
                               "for 32×32 CIFAKE-style images use one of the 32×32 models.", icon="⚠️")
            elif min(img.size) > 64:
                rate = unseen_detection_rate(run)
                st.error(
                    f"**Unreliable for this image.** It is {img.size[0]}×{img.size[1]} px, but the model was trained "
                    "only on 32×32 CIFAKE images (CIFAR-10 objects vs. Stable Diffusion v1.4). Shrinking it to 32×32 "
                    "removes the fine detail the detector relies on, and images from other generators or with new "
                    "content (e.g. portraits) are outside its training data."
                    + (f" On our test of 6 unseen generators (incl. Midjourney) this model detected only "
                       f"**{rate:.0f}%** of AI images." if rate is not None else ""),
                    icon="⚠️")

            pic, details = st.columns([3, 2])
            pic.image(heatmap_on_image(img, cam), width="stretch",
                      caption=f"{name} ({img.size[0]}×{img.size[1]}). Heatmap of the regions behind the model's "
                              f"score (red = strongest), drawn on the square it analysed; the dimmed parts were not analysed.")
            with details.expander("What the file says about itself", expanded=True):
                if face is None:
                    st.markdown("- **Portrait check:** no face found.")
                else:
                    st.markdown(f"- **Portrait check:** {face['faces']} face(s); the largest is {face['face_size']:.0%} of "
                                f"the image's shorter side, "
                                + (f"inside the portrait range ({ZONE[0]:.0%}–{ZONE[1]:.0%}) where the model has failed "
                                   "on real-world photos." if face["in_zone"] else "outside the portrait range."))
                cr = evidence["c2pa"]
                if cr is None:
                    st.markdown("- **Content Credentials (C2PA):** none in this file.")
                else:
                    state = {"Trusted": "signature intact, signer on the official trust list",
                             "Valid": "signature intact" + (", signer not on the official trust list" if cr["trust_list"]
                                                            else " (trust list not installed)"),
                             "Invalid": "verification failed: the file was changed after signing"}.get(cr["state"], cr["state"])
                    what = cr["generated"] or ("camera capture" if cr["capture"] else "no origin stated")
                    st.markdown(f"- **Content Credentials (C2PA):** {what}; signed by {cr['signer'] or 'unnamed signer'}"
                                + (f" with {cr['tool']}" if cr["tool"] else "") + f" — {state}.")
                c = evidence["camera"]
                if c:
                    st.markdown(f"- **Camera metadata:** {c['make']} {c['model']}"
                                + (f", {c['lens']}" if c["lens"] else "") + (f", taken {c['taken']}" if c["taken"] else "")
                                + ". Supports a real photo, but is weak evidence: it can be copied onto a fake.")
                for reason in evidence["ai_declared"]:
                    st.markdown(f"- **AI declaration:** {reason}.")
                for note in evidence["notes"]:
                    st.markdown(f"- {note[0].upper() + note[1:]}.")
            st.caption("A detector's output is evidence, not proof. Wrongly labelling a real photo as fake can harm "
                       "people — report results with care.")

with tab_results:
    summary = RES / "summary.csv"
    if summary.exists():
        st.subheader("Summary of all models")
        st.dataframe(list(csv.DictReader(open(summary, encoding="utf-8"))), width="stretch", hide_index=True)
    rows = v2_rates()
    if rows:
        st.subheader("v2 · 224×224 detector for large images (share classified correctly, clean images)")
        st.dataframe([{"group": r["group"], "source": r["source"], "type": r["label"], "images": r["n"],
                       "correct %": f"{100 * float(r['correct_rate']):.1f}"} for r in rows],
                     width="stretch", hide_index=True)
        if (FIG / "v2_results.png").exists():
            st.image(str(FIG / "v2_results.png"), width="stretch")
    sections = [
        ("O1 · Accuracy on CIFAKE", ["training_curves.png", "confusion_matrices.png"]),
        ("O2 · Unseen generators (GenImage)", ["genimage_detection.png", "genimage_balanced_accuracy.png"]),
        ("O3 · Robustness to compression, resizing and blur", ["robustness_f1.png"]),
        ("O4 · What the models look at", ["../spectrum.png"]),
    ]
    for title, files in sections:
        present = [FIG / f for f in files if (FIG / f).exists()]
        if present:
            st.subheader(title)
            for f in present:
                st.image(str(f), width="stretch")
    cams = sorted(RES.glob("gradcam_*_examples.png"))
    if cams:
        pick = st.selectbox("Grad-CAM examples", [c.stem.replace("gradcam_", "").replace("_examples", "") for c in cams])
        st.image(str(RES / f"gradcam_{pick}_examples.png"), width="stretch")
    for f in ["summary.csv", "robustness.csv", "genimage.csv"]:
        if (RES / f).exists():
            st.download_button(f"Download {f}", (RES / f).read_bytes(), file_name=f)

with tab_about:
    readme = ROOT / "README.md"
    st.markdown(readme.read_text(encoding="utf-8") if readme.exists() else "See README.md")
