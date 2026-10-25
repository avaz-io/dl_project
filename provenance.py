"""Read what an image file says about its own origin, before any model looks at the pixels.

    from provenance import inspect, combine
    ev = inspect(file_bytes)                 # evidence found in the file
    verdict = combine(ev, model_score)       # merge it with the detector's score (0-1)

Kinds of evidence, from strongest to weakest:
  1. Content Credentials (C2PA): a record signed by the camera or AI tool that made the file,
     verified with the official c2pa-python library (signature, and the signer against the
     Content Credentials trust list when it is installed: python provenance.py --update-trust-list).
     A trusted record saying "AI-generated" or "camera capture" is the strongest evidence here.
  2. Other declarations that the image is AI-generated: IPTC "digital source type" metadata, or
     generation settings embedded by Stable Diffusion tools.
  3. An AI watermark found by an external watermark check (passed in by the app).
  4. Plain camera metadata (make, model, exposure). It supports "real photo" but is WEAK evidence:
     it is easy to copy onto a fake, and messengers strip it. Its absence proves nothing.

None of this is proof on its own, and all of it is lost when a file is screenshotted or re-shared.
"""
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

AI_SOURCE_TYPES = [b"trainedAlgorithmicMedia", b"compositeWithTrainedAlgorithmicMedia", b"algorithmicMedia"]
AI_TOOLS = ["midjourney", "dall-e", "dall·e", "dalle", "stable diffusion", "firefly", "novelai", "comfyui", "imagen",
            "flux", "leonardo", "ideogram", "automatic1111", "invokeai", "fooocus", "playground"]
SD_TEXT_KEYS = {"parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata", "dream", "generation_data"}
CAPTURE_TAGS = ["ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "LensModel", "DateTimeOriginal"]

# Content Credentials (C2PA)
C2PA_DIR = Path(__file__).parent / "data" / "c2pa"
TRUST_URL = "https://contentcredentials.org/trust/"
TRUST_FILES = {"trust_anchors": "anchors.pem", "allowed_list": "allowed.sha256.txt", "trust_config": "store.cfg"}
GENERATED_TYPES = {"trainedAlgorithmicMedia": "AI-generated", "compositeWithTrainedAlgorithmicMedia": "partly AI-generated",
                   "algorithmicMedia": "computer-generated"}  # IPTC digital source types
CAPTURE_TYPES = {"digitalCapture", "computationalCapture", "negativeFilm", "positiveFilm", "print"}
MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "TIFF": "image/tiff", "HEIF": "image/heif",
        "AVIF": "image/avif", "GIF": "image/gif"}
_c2pa_context = None


def update_trust_list():
    """Download the official Content Credentials trust list into data/c2pa/."""
    C2PA_DIR.mkdir(parents=True, exist_ok=True)
    for name in TRUST_FILES.values():
        urllib.request.urlretrieve(TRUST_URL + name, C2PA_DIR / name)
    print(f"trust list saved to {C2PA_DIR}")


def _context():
    """c2pa Context that checks signers against the trust list, or None if the list is not installed."""
    global _c2pa_context
    if _c2pa_context is None:
        import c2pa
        if all((C2PA_DIR / f).exists() for f in TRUST_FILES.values()):
            trust = {k: (C2PA_DIR / f).read_text(encoding="utf-8") for k, f in TRUST_FILES.items()}
            _c2pa_context = c2pa.Context(c2pa.Settings.from_dict({"trust": trust, "verify": {"verify_trust": True}}))
        else:
            _c2pa_context = False
    return _c2pa_context or None


def c2pa_check(data, fmt=None):
    """Read and verify Content Credentials. Returns None when the file has no C2PA record (or the
    c2pa-python library is missing), else a dict:
      state     "Trusted" (signature intact, signer on the trust list), "Valid" (intact, signer not
                verified as trusted) or "Invalid" (the file or record was changed after signing)
      generated "AI-generated" / "partly AI-generated" / "computer-generated", or None
      capture   True if the record says the image was captured by a camera
      signer, tool, failures (validation codes), trust_list (whether the trust list was used)
    """
    try:
        import c2pa
    except ImportError:
        return None
    try:
        reader = c2pa.Reader.try_create(MIME.get(fmt), io.BytesIO(data), context=_context())
    except Exception as e:  # unreadable or malformed record
        return {"state": "Invalid", "generated": None, "capture": False, "signer": "", "tool": "",
                "failures": [f"unreadable record ({type(e).__name__})"], "trust_list": _context() is not None}
    if reader is None:
        return None
    try:
        store = json.loads(reader.json())
    finally:
        reader.close()
    active = store.get("manifests", {}).get(store.get("active_manifest"), {})
    sources = []
    for a in active.get("assertions", []):
        if a.get("label", "").startswith("c2pa.actions"):
            sources += [str(act.get("digitalSourceType", "")) for act in a.get("data", {}).get("actions", [])]
    kinds = [GENERATED_TYPES[k] for k in GENERATED_TYPES for t in sources if t.endswith("/" + k)]
    info = active.get("claim_generator_info") or []
    tool = (info[0].get("name", "") if info and isinstance(info[0], dict) else "") or active.get("claim_generator", "")
    sig = active.get("signature_info") or {}
    failures = [f.get("code", "") for f in (store.get("validation_results") or {}).get("activeManifest", {}).get("failure", [])]
    return {"state": store.get("validation_state") or ("Invalid" if failures else "Valid"),
            "generated": kinds[0] if kinds else None,
            "capture": any(t.rsplit("/", 1)[-1] in CAPTURE_TYPES for t in sources),
            "signer": sig.get("issuer") or sig.get("common_name") or "", "tool": str(tool)[:60],
            "failures": failures, "trust_list": _context() is not None}


def inspect(data):
    """Return {"ai_declared": [reasons], "camera": {...} or None, "camera_verified": {...} or None,
    "c2pa": {...} or None, "notes": [...]} for raw file bytes."""
    ev = {"ai_declared": [], "camera": None, "camera_verified": None, "c2pa": None, "notes": []}
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        im = None

    cr = ev["c2pa"] = c2pa_check(data, im.format if im else None)
    if cr:
        who = f"signed by {cr['signer'] or 'an unnamed signer'}" + (f", made with {cr['tool']}" if cr["tool"] else "")
        trust = ("signer is on the official trust list" if cr["state"] == "Trusted" else
                 "signer not on the official trust list" if cr["trust_list"] else "trust list not installed")
        if cr["state"] == "Invalid":
            reasons = [f for f in cr["failures"] if f != "signingCredential.untrusted"] or cr["failures"]
            ev["notes"].append(f"Content Credentials found but verification FAILED ({', '.join(reasons[:3])}): "
                               f"the file or its record was changed after signing, so its claims are ignored")
        elif cr["generated"]:
            ev["ai_declared"].append(f"verified Content Credentials say {cr['generated']} ({who}; {trust})")
        elif cr["capture"] and cr["state"] == "Trusted":
            ev["camera_verified"] = {"signer": cr["signer"], "tool": cr["tool"]}
        else:
            ev["notes"].append(f"Content Credentials found ({who}; {trust})"
                               + ("; they claim a camera capture, but only a trusted signer makes that strong evidence"
                                  if cr["capture"] else ""))
    else:  # no C2PA record: fall back to a plain scan for IPTC digital-source-type metadata
        for marker in AI_SOURCE_TYPES:
            if marker in data:
                ev["ai_declared"].append(f"metadata declares digital source type '{marker.decode()}'")
                break

    if im is None:
        ev["notes"].append("file could not be parsed for metadata")
        return ev

    # Generation settings written by Stable Diffusion front-ends into PNG text chunks
    text = {k.lower(): v for k, v in im.info.items() if isinstance(v, str)}
    for key in sorted(SD_TEXT_KEYS & set(text)):
        if key != "prompt" or len(text[key]) > 20:
            ev["ai_declared"].append(f"embedded generation settings ('{key}' field)")
            break

    exif = im.getexif()
    tags = {TAGS.get(k, k): v for k, v in exif.items()} if exif else {}
    sub = {TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()} if exif else {}
    xmp_tool = re.search(rb"CreatorTool[>=\"' ]+([^<\"']{2,60})", data)
    software = " ".join(str(x) for x in [tags.get("Software", ""), text.get("software", ""),
                                         xmp_tool.group(1).decode("latin-1") if xmp_tool else ""]).strip()
    hit = next((t for t in AI_TOOLS if t in software.lower()), None)
    if hit:
        ev["ai_declared"].append(f"software field names an AI tool ('{software[:50]}')")
    elif software and any(s in software.lower() for s in ("photoshop", "lightroom", "gimp", "snapseed")):
        ev["notes"].append(f"edited with {software[:40]}")

    make, model = str(tags.get("Make", "")).strip(), str(tags.get("Model", "")).strip()
    capture = {k: sub[k] for k in CAPTURE_TAGS if k in sub}
    if make and model and capture:
        ev["camera"] = {"make": make, "model": model, "lens": str(capture.get("LensModel", ""))[:60],
                        "taken": str(capture.get("DateTimeOriginal", tags.get("DateTime", ""))),
                        "fields": len(capture)}
    elif make or model:
        ev["notes"].append(f"partial camera metadata ({make} {model}), no exposure details".replace("  ", " "))
    if not (ev["ai_declared"] or ev["camera"] or ev["camera_verified"] or ev["notes"]):
        ev["notes"].append("no origin metadata in the file (common: most apps and websites strip it)")
    return ev


def combine(ev, p_ai, watermark=None, portrait=None):
    """Merge file evidence with the model's AI score (0-1). Returns (kind, headline, explanation);
    kind is one of 'ai_declared', 'real_verified', 'ai', 'real', 'inconclusive'.
    watermark: True if an external watermark check found an AI watermark.
    portrait: result of portrait.portrait_zone(); a face in the unreliable size range makes a
    model-only verdict inconclusive (file evidence still decides)."""
    if watermark:
        return ("ai_declared", "AI-generated (watermark found)",
                "An AI watermark was found in this image. This does not depend on the model.")
    if ev["ai_declared"]:
        return ("ai_declared", "AI-generated (declared by the file)",
                f"The file itself says so: {ev['ai_declared'][0]}. This does not depend on the model.")
    vc = ev.get("camera_verified")
    if vc:
        disagree = " The model disagrees (it scores the image as AI), which is a known weakness." if p_ai >= 0.5 else ""
        return ("real_verified", "Real photo (verified camera signature)",
                f"Content Credentials signed by {vc['signer']} (on the official trust list) say this image was "
                f"captured by a camera, and the file has not been changed since.{disagree}")
    cam = ev["camera"]
    if portrait and portrait.get("in_zone"):
        said = f"The model's score is {100 * p_ai:.1f}% AI-generated, but"
        return ("inconclusive", "Inconclusive: portraits are a known weak spot",
                f"{said} this image shows a person at portrait distance. In spot checks with real-world "
                "portraits (two real phone photos and one photorealistic AI portrait) the model was wrong all "
                "three times, in both directions. On the benchmark's portraits it is right about 88% of the "
                "time, so the score may be correct, but do not rely on it."
                + (f" The file also carries camera metadata ({cam['make']} {cam['model']})." if cam else ""))
    if p_ai >= 0.5 and cam:
        return ("inconclusive", "Inconclusive: the evidence conflicts",
                f"The model scores this image as AI-generated, but the file carries camera metadata "
                f"({cam['make']} {cam['model']}). The model is known to raise false alarms on some real photos, "
                "and camera metadata can be copied onto a fake, so neither settles it.")
    if p_ai >= 0.5:
        return ("ai", "Model says: AI-generated", "No origin metadata in the file contradicts or confirms this.")
    if cam:
        return ("real", "Model says: Real photo",
                f"The file also carries camera metadata ({cam['make']} {cam['model']}), which supports this "
                "(weak evidence: metadata can be faked).")
    return ("real", "Model says: Real photo", "No origin metadata in the file contradicts or confirms this.")


if __name__ == "__main__":
    if "--update-trust-list" in sys.argv:
        update_trust_list()
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        print(p, json.dumps(inspect(Path(p).read_bytes()), indent=2, default=str))
