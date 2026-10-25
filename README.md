# Real or AI? Detecting AI-generated images with lightweight CNNs

Deep Learning course project. We train CNNs to tell real photos from AI-generated images on
**CIFAKE** (60k real CIFAR-10 photos vs. 60k Stable Diffusion v1.4 images, 32×32), then test
how far the detectors can be trusted:

- **O1 – Accuracy** on the CIFAKE test set (20,000 images)
- **O2 – Generalization** to images from generators never seen in training (GenImage)
- **O3 – Robustness** to JPEG compression, resizing, blur and messenger-style re-sharing
- **O4 – Explanation**: frequency analysis and Grad-CAM

## Main results

| Model | Training | Params | Train time | CIFAKE acc. | F1 | ROC-AUC | F1 after "messenger" re-share |
|---|---|---|---|---|---|---|---|
| Baseline CNN (CIFAKE paper: 92.9%) | standard | 0.08M | 1 min | 94.9% | 94.9% | 0.989 | 33.6% |
| Baseline CNN | compression-aware | 0.08M | 3 min | 94.3% | 94.4% | 0.988 | 90.0% |
| ResNet-18 | standard | 11.2M | 11 min | 98.3% | 98.3% | 0.999 | 33.0% |
| ResNet-18 | compression-aware | 11.2M | 12 min | 98.0% | 98.0% | 0.999 | 96.4% |
| EfficientNet-B0 | standard | 4.0M | 24 min | **98.9%** | **98.9%** | 0.999 | 42.9% |
| **EfficientNet-B0** | **compression-aware** | 4.0M | 25 min | 98.6% | 98.6% | 0.999 | **96.9%** |

All models trained on a laptop RTX 2050 (4 GB). Full numbers: `results/summary.md`.

**Key findings**

1. Pretrained CNNs reach 98–99% on CIFAKE, well above the paper's 92.9% baseline.
2. **Standard detectors are fragile.** JPEG compression barely matters, but downscaling by 25%
   or a light blur drops F1 to 33–51%: the models then call almost every AI image "real"
   (only 20–27% of AI images caught after messenger-style re-sharing).
3. **Why:** AI images in CIFAKE carry ~1.35× more high-frequency energy than real photos.
   Blur/downscaling removes it (to 0.2–0.5× of real), so the fingerprint flips
   (`results/spectrum.png`).
4. **Compression-aware training fixes it:** with random JPEG/rescale/blur during training, F1
   stays ≥ 95% under every degradation (≥ 94.5% for ResNet-18 at the harshest), at a cost of only
   0.3–0.5 points on clean images.
5. Note: CIFAKE images are already JPEG quality 75, so a "JPEG 75" test changes almost nothing;
   JPEG 50/30 and the resize/blur conditions are the meaningful robustness tests.

### O2 – Unseen generators (GenImage)

Test set: 300 AI images from each of 6 generators never seen in training (ADM, BigGAN, GLIDE,
Midjourney, VQDM, Wukong) + 1,123 real ImageNet photos, all converted to CIFAKE format.

| Model | Training | ADM | BigGAN | GLIDE | Midjourney | VQDM | Wukong | **All** | Real photos correct | Balanced acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline CNN | standard | 9% | 18% | 32% | 44% | 30% | 77% | **35%** | 71% | 53.2% |
| Baseline CNN | compression-aware | 3% | 14% | 34% | 38% | 25% | 76% | **32%** | 77% | 54.5% |
| ResNet-18 | standard | 3% | 12% | 13% | 33% | 26% | 76% | **27%** | 83% | 54.8% |
| ResNet-18 | compression-aware | 1% | 7% | 9% | 27% | 24% | 68% | **23%** | 86% | 54.6% |
| EfficientNet-B0 | standard | 2% | 9% | 8% | 28% | 27% | 71% | **24%** | 84% | 54.0% |
| EfficientNet-B0 | compression-aware | 1% | 12% | 5% | 19% | 26% | 65% | **21%** | 86% | 53.7% |

(Generator columns = share of that generator's AI images detected.)

6. **The detectors do not generalize to unseen generators.** Balanced accuracy is 53–55%, close
   to chance, and only 21–35% of AI images are caught. ROC-AUC is ~0.55 overall and *below* 0.5
   for ADM (0.28–0.32): ADM images look *more* "real" to the models than real photos do.
7. **The one exception is Wukong (65–77% detected)**, a latent-diffusion model from the same
   family as Stable Diffusion v1.4, which made the CIFAKE fakes. The detectors learn the
   fingerprint of *one generator family*, not a general notion of "AI-generated" — consistent
   with GenImage's own cross-generator results and with Wang et al. / Ojha et al.
8. Real ImageNet photos are also harder: 14–29% are wrongly flagged (vs. 1.5–7% on CIFAKE).
9. Compression-aware training does **not** help here (O2 target of +5 points not met): it
   fixes robustness to *processing*, not to *new generators*. Closing that gap needs training
   on several generators or generator-agnostic features (e.g. CLIP features, as in Ojha et al.).

Figures: `results/figures/genimage_detection.png`, `genimage_balanced_accuracy.png`.

## Extension: v2 detector for large, real-world images

The CIFAKE models only see 32×32 images from one generator, so they fail on real-world uploads
(a 1138×1280 AI portrait was called "real"). v2 is an EfficientNet-B0 trained at **224×224** with
the compression-aware augmentation on real photos and AI images from seven generators (SDXL faces
and scenes, RealVisXL, JourneyDB/Midjourney, GenImage ADM, VQDM, Wukong). 12 epochs, 26 min on the
RTX 2050. Two versions were trained; they differ only in the real photos used:

- **v2a** – real photos: FFHQ and CelebA-HQ faces, MS-COCO scenes (28,811 images in total)
- **v2b** – v2a plus real object photos from Caltech-256 and Open Images (37,733 images); v2c and v2d (current) add more real photos, see below

Held-out sources, never seen in training (share classified correctly, clean images):

| Held-out source | 32×32 models | v2a | **v2b** |
|---|---|---|---|
| Real: ImageNet photos | 71–86% | 48% | **87%** |
| AI: Midjourney (GenImage) | 19–44% | 100% | **97%** |
| AI: FLUX faces | – | 94% | **83%** |
| AI: FLUX scenes | – | 83% | **73%** |
| AI: GLIDE | 5–34% | 97% | **71%** |
| AI: BigGAN | 7–18% | 91% | **43%** |
| AI detected, all held-out generators | 21–35% | 92% | **70%** |
| **Balanced accuracy (held-out)** | 53–55% | 70% | **79%** |

On the sources it trained on (test split), v2b gets 95–100% of AI images and 93–100% of real
photos right, and stays within about 5 points after messenger-style re-sharing. (The 32×32 column
is from the O2 test: the same generators, a different random sample, at 32×32.)

- **v2a looked better than it was.** It caught 92% of AI images from unseen generators, but called
  half of real ImageNet photos AI. Its training set had object/animal images almost only on the AI
  side (GenImage is ImageNet-class content) and real photos of faces and scenes only, so it could
  use *content* as a shortcut: "object or animal → AI".
- **v2b removes that shortcut.** Adding real object photos cut false alarms on real ImageNet photos
  from 52% to 13% and raised held-out balanced accuracy from 70% to 79%. Detection of unseen
  generators fell to 70% (BigGAN 43%, GLIDE 71%), which is the honest number: much of v2a's
  "detection" of those object-centric fakes came from the content shortcut.
- **Compared with the 32×32 models**, v2b detects 70% instead of 21–35% of AI images from unseen
  generators, and flags the AI portrait that fooled v1 (score 100%).
- **Still not reliable for unknown generators** (BigGAN: 43%). Closing that gap needs more
  generators in training, or generator-agnostic features such as CLIP (Ojha et al.).
- **Lesson for the report:** a detector must be tested on held-out *real* photos as well as
  held-out generators; a high detection rate alone can hide a model that calls everything AI.

- **Two bugs found and fixed on the way** (both worth a line in the report):
  1. Image tensors were left in NHWC memory order after `permute`; with mixed precision at 224×224
     this made training 13× slower (2.7 s → 0.2 s per step after `.contiguous()`).
  2. Zoom augmentation applied to *every* batch slightly blurred all training images, so sharp,
     untouched validation photos looked "AI" (62% correct on real MS-COCO). Applying the zoom to
     half the batches raised validation balanced accuracy from 90–94% to 99%.

Figure: `results/figures/v2_results.png`. Numbers: `results/v2_results.csv`.

```bash
python download_v2.py            # ~4.3 GB, one parquet shard per source (resumable)
python build_v2_dataset.py       # 224x224 dataset -> data/v2/
python train_v2.py --epochs 12   # add --resume auto to continue an interrupted run
python predict.py photo.jpg      # check any image with the newest v2 model
```

### v2c, v2d and a false alarm that three data patches did not fix

A real iPhone 14 Pro Max photo (3024×4032, a person sitting in front of mountains) is scored
**100% "AI"** by every v2 model. Two more retrainings targeted it:

| Model | Real photos added | Held-out real correct | Held-out AI detected | Balanced | The iPhone photo |
|---|---|---|---|---|---|
| v2b | Caltech-256, Open Images | 88.6% | 74.6% | 81.6% | AI, 99.9% |
| v2c | + LFW people, Open Images high-res | 90.9% | 73.0% | 82.0% | AI, 100% |
| **v2d (current)** | + 4,500 MS-COCO photos with a prominent person (full photos and chest-up to full-body crops) | **94.0%** | 68.8% | 81.4% | AI, 100% |

(Held-out real = 631 full-resolution camera photos + 1,123 ImageNet photos; held-out AI = FLUX,
GLIDE, Midjourney and BigGAN images. The held-out real photos are identical for all three models; the
held-out AI images are re-sampled at each rebuild, which moved v2c's figure by only 0.2 points.)

Each patch lowered false alarms on held-out real photos (11% → 9% → 6%) and cost some detection
of unseen generators; balanced accuracy stayed at about 82%. None changed the verdict on this
photo. What the measurements show (an occlusion heatmap was made but is not in the repository,
because it shows a real person):

- **It is the person, not the image quality or the scene.** Hiding the background, sky or torso
  leaves the score at 100%; hiding the whole person drops it to 42%. Blur, JPEG compression,
  resizing, grayscale, mirroring and moving the person to the edge of the frame change nothing.
- **Scale matters.** Zoomed in until the face fills the frame, the same face is scored 0% AI. At
  normal distance (face about 35 px wide in the 224×224 input) it is scored AI.
- **Two hypotheses were tested and rejected.** "Any real person with a visible torso is called
  AI" is false: v2c already classified 96.7% of real COCO people photos correctly before they
  were added to training. "Sharp, high-resolution camera photos are called AI" is false: 94–96%
  of held-out full-resolution camera photos are classified correctly.
- **Not explained.** Which training images taught the model this is unknown. A mistake made on
  the way: the second and third patches were chosen from a plausible story and only afterwards
  checked against the existing model. Scoring the existing model on the candidate data *first*
  would have shown that the third patch could not help.

**Conclusions for the report**

1. A detector's score is not a confidence: the model is wrong here at 100%.
2. Aggregate accuracy (94% of held-out real photos correct) does not protect an individual
   image; the errors are systematic, not random.
3. Adding data closes the gaps you can name, not the ones you cannot.
4. This is why the app reads the file's own metadata first and reports "Inconclusive" when a
   model verdict of AI conflicts with camera metadata (next section).

## File evidence first: provenance check before the model

A pixel-based detector can be confidently wrong (v2 scored a real iPhone photo 100% "AI"). So the
app first reads what the file says about its own origin (`provenance.py`) and only then asks the
model:

| Step | Evidence | Effect on the verdict |
|---|---|---|
| 1 | **Content Credentials (C2PA)**, verified with the official `c2pa-python` library: a record signed by the camera or AI tool that made the file | AI-generated record → "AI-generated (declared by the file)"; camera-capture record from a signer on the official trust list → **"Real photo (verified camera signature)"**; record broken by later changes → ignored, with a warning |
| 1b | Other declared AI origin: IPTC "digital source type", Stable Diffusion settings embedded in the file, an AI tool in the software field | "AI-generated (declared by the file)", whatever the model says |
| 2 | AI watermark found by the optional watermark check on a second PC (see below) | same as step 1 |
| 3 | Camera metadata (make, model, exposure) | model says AI + camera metadata → **"Inconclusive: the evidence conflicts"** |
| 4 | The model's score | used as before when the file says nothing |

- Camera metadata is **weak** evidence: it is easy to copy onto a fake, and messengers strip it.
  Its absence proves nothing, so it never turns a verdict into "real"; it only stops the app from
  asserting "AI-generated" when the evidence conflicts.
- Most AI images online carry no declaration (the AI portrait used in testing has no metadata at
  all), so steps 1–2 rarely fire. This makes the app more honest, not the model more accurate.
- **Content Credentials** are checked cryptographically: the signature must match the file, and
  the signer is checked against the official Content Credentials trust list (download it with
  `python provenance.py --update-trust-list`; without it, records can only be reported as "signature
  intact", not "trusted"). Only a *trusted* camera record counts as strong evidence of a real photo,
  because anyone can sign a record with their own certificate. Records are added by e.g. ChatGPT/DALL·E,
  Google Gemini, Pixel 10 cameras and some Leica, Sony, Canon and Fujifilm cameras, but they are lost
  in screenshots and most messengers and social networks, so most images will not have one.
- Tested with the official C2PA test image (`tests/fixtures/C.jpg`, from the c2pa-python repository)
  and a tampered copy of it. A real record from a trusted camera or AI service has not been tested yet.

**Portrait safeguard** (`portrait.py`). In spot checks with real-world portraits the v2 model was
wrong 3 times out of 3, in both directions: two real phone portraits scored 98–100% "AI", and a
photorealistic AI portrait scored 0.6% (the first v2 model caught it; the real-photo patches taught
the model to pass it). So when the OpenCV YuNet face detector finds a face of portrait size (8–55%
of the image's shorter side) and the file itself proves nothing, the app shows **"Inconclusive:
portraits are a known weak spot"** with the model's score, instead of a confident verdict.

The benchmark does *not* show this weakness, and the safeguard has a measured cost
(`python portrait.py --evaluate`, test and held-out images):

| Largest face | Images | Real correct | AI correct | Balanced |
|---|---|---|---|---|
| no face | 7,020 | 95.3% | 73.7% | 84.5% |
| 8–20% | 773 | 98.3% | 77.6% | 88.0% |
| 20–35% | 913 | 98.3% | 81.6% | 90.0% |
| 35–55% | 1,107 | 99.6% | 60.6% | 80.1% |
| > 55% (close-ups) | 495 | 100.0% | 57.1% | 78.6% |

Inside the safeguard's range the model is right 87.8% of the time (84.1% outside), and the safeguard
fires on 26.7% of these images. The benchmark's portraits come from the same kinds of datasets the
model trained on; real-world portraits (phone photos, web images from unknown generators) do not,
and those are where it failed. The safeguard therefore trades away many correct verdicts (it also
turned a correctly detected AI illustration of a boy into "Inconclusive") to avoid confident wrong
ones on real-world portraits, on the evidence of only three spot checks. Measuring it properly needs
a held-out set of real-world portraits, real and AI.

```bash
python provenance.py --update-trust-list   # once: official Content Credentials trust list -> data/c2pa/
python test_provenance.py                  # checks every verdict rule, including C2PA, on test files
```

## Watermark check on a second PC (optional)

When `DESK_URL` and `DESK_TOKEN` are set in `.env`, the app sends every image it analyses to a
watermark-check service on a second PC (over Tailscale or a private network) and shows that
check's answer first, before its own verdict: "watermark found" makes the verdict "AI-generated
(watermark found)"; "no watermark" changes nothing, because most AI tools add none. While the check
is running, the verdict is marked as preliminary and updates by itself. The app tells users that
images are sent to an external check. The service on the second PC is not part of this repository.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

Model weights: the repository includes `best.pt` for the six 32×32 models in `runs/` and for
the current v2 model (`runs_v2/efficientnet_b0_224_20261013_190407`), so the app works without
training. The other v2 runs keep only their `results.json` and `history.json`.

Data:
- **CIFAKE**: download from Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`)
  and unzip into `data/cifake/` (`train/REAL`, `train/FAKE`, `test/REAL`, `test/FAKE`).
  The first run of `train.py` caches all images to `data/cifake_cache.npz`.
- **GenImage** (O2 only): AI images come from one parquet shard per generator of the BitMind
  Hugging Face mirror (`bitmind/GenImage_<Generator>`) saved as `data/genimage/hf/<Generator>.parquet`;
  real ImageNet photos come from the official GenImage Google Drive index parts
  `imagenet_ai_0508_adm.zip` and `imagenet_glide.zip` in `data/genimage/raw/`
  (`genimage_index.py` reads them without the other 3 GB parts).

## Reproduce

```bash
python check_data.py                                   # sanity-check CIFAKE, sample grid
python train.py --model baseline --epochs 15           # O1
python train.py --model resnet18 --epochs 10
python train.py --model efficientnet_b0 --epochs 10
python train.py --model resnet18 --epochs 10 --compress-aug          # O3 (also baseline / efficientnet_b0)
python robustness.py                                   # O3: F1 under 10 degradations -> results/robustness.csv
python build_genimage_test.py && python evaluate_genimage.py         # O2 -> results/genimage.csv
python explain.py spectrum && python explain.py gradcam               # O4
python make_figures.py                                 # figures + summary tables
.venv\Scripts\streamlit run app.py --server.address localhost         # demo app
```

## Files

| File | Purpose |
|---|---|
| `train.py` | data loading/cache, models (baseline CNN, ResNet-18, EfficientNet-B0), training, test metrics |
| `degrade.py` | JPEG / rescale / blur / messenger degradations (shared by training and testing) |
| `robustness.py` | O3 evaluation |
| `genimage_index.py`, `build_genimage_test.py`, `evaluate_genimage.py` | O2 test set and evaluation |
| `explain.py` | O4: frequency spectrum and Grad-CAM |
| `make_figures.py` | report figures (`results/figures/`) and `results/summary.*` |
| `app.py` | Streamlit app: watermark check, then file evidence and model, with the heatmap drawn on the image + results dashboard |
| `provenance.py`, `test_provenance.py` | file-origin evidence (Content Credentials, AI declarations, camera metadata) and its tests |
| `watermark_client.py` | sends each image to the optional watermark check on a second PC and reads the answer |
| `portrait.py`, `models/face_detection_yunet_2023mar.onnx` | portrait safeguard: face detection (OpenCV YuNet, from opencv_zoo) and its evaluation |
| `download_coco_people.py`, `build_coco_people.py` | real photos with a prominent person (chest-up to full body) from annotated MS-COCO |
| `run_v2_fix.py`, `eval_v2_run.py` | detached retrain-and-compare pipeline; evaluates an older model on the current dataset |
| `predict.py`, `test_app.py` | check image files from the command line; headless app test |
| `notify.py` | Telegram notifications (reads `.env`; never share or commit that file) |
| `runs/<model>[_compress]_<time>/` | `best.pt`, `history.json`, `results.json` per training run |

## Limitations

- Models see 32×32 images; large real-world photos are shrunk first, which removes detail.
- A detector's output is evidence, not proof — always report the false-positive rate.
