"""Checks for provenance.py: synthetic files with each kind of metadata, plus any image files given.

    python test_provenance.py [image files to inspect...]
"""
import io
import sys
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from provenance import combine, inspect


def jpeg_with_camera():
    im = Image.new("RGB", (64, 64), "gray")
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "TestCam", "Model X"  # Make, Model
    sub = exif.get_ifd(0x8769)
    sub[0x829A], sub[0x829D], sub[0x8827] = (1, 250), (28, 10), 100  # ExposureTime, FNumber, ISO
    b = io.BytesIO()
    im.save(b, "JPEG", exif=exif)
    return b.getvalue()


def png(text=None):
    info = PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(b, "PNG", pnginfo=info)
    return b.getvalue()


def check(name, data, p_ai, want_kind):
    ev = inspect(data)
    kind, headline, _ = combine(ev, p_ai)
    ok = kind == want_kind
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} model {100 * p_ai:5.1f}% -> {headline}")
    return ok


def main():
    cam, plain = jpeg_with_camera(), png()
    sd = png({"parameters": "a portrait photo, Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 1"})
    iptc = plain + b"<x:xmpmeta>Iptc4xmpExt:DigitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</x:xmpmeta>"
    tool = png({"Software": "Midjourney v6"})
    results = [
        check("camera metadata, model says AI", cam, 0.999, "inconclusive"),
        check("camera metadata, model says real", cam, 0.02, "real"),
        check("no metadata, model says AI", plain, 0.97, "ai"),
        check("no metadata, model says real", plain, 0.10, "real"),
        check("Stable Diffusion settings, model says real", sd, 0.05, "ai_declared"),
        check("IPTC 'trainedAlgorithmicMedia', model says real", iptc, 0.05, "ai_declared"),
        check("software names an AI tool, model says real", tool, 0.30, "ai_declared"),
        check("camera metadata + AI declaration", cam + b"trainedAlgorithmicMedia", 0.5, "ai_declared"),
    ]
    # Content Credentials: the official c2pa-python test image (signed with a test certificate)
    signed = (Path(__file__).parent / "tests" / "fixtures" / "C.jpg").read_bytes()
    tampered = bytearray(signed)
    tampered[-5000] ^= 0xFF  # one changed byte in the image data
    results += [
        check("C2PA record declaring computer-generated", signed, 0.05, "ai_declared"),
        check("C2PA record, file changed after signing", bytes(tampered), 0.05, "real"),
    ]
    cr = inspect(signed)["c2pa"]
    assert cr["state"] in ("Valid", "Trusted") and cr["generated"] == "computer-generated", cr
    assert cr["signer"] == "C2PA Test Signing Cert", cr
    bad = inspect(bytes(tampered))
    assert bad["c2pa"]["state"] == "Invalid" and not bad["ai_declared"], "a broken record must not count as a declaration"
    assert any("dataHash" in n for n in bad["notes"]), bad["notes"]
    assert inspect(cam)["c2pa"] is None and inspect(plain)["c2pa"] is None
    # verdict rule for a trusted camera signature (no real camera-signed file is bundled, so test the rule directly)
    ev = {"ai_declared": [], "camera": None, "notes": [], "c2pa": None,
          "camera_verified": {"signer": "Example Camera CA", "tool": "Example Camera"}}
    kind, headline, why = combine(ev, 0.99)
    print(f"{'PASS' if kind == 'real_verified' else 'FAIL'}  {'trusted camera signature, model says AI':46s} model  99.0% -> {headline}")
    results.append(kind == "real_verified" and "disagrees" in why)

    # portrait safeguard: a face in the unreliable range overrides model-only verdicts, not file evidence
    face_in = {"faces": 1, "face_size": 0.2, "in_zone": True}
    face_out = {"faces": 1, "face_size": 0.7, "in_zone": False}
    for name, data, p, face, want in [("portrait, model says AI", plain, 0.99, face_in, "inconclusive"),
                                      ("portrait, model says real", plain, 0.01, face_in, "inconclusive"),
                                      ("close-up face (outside range), model says AI", plain, 0.99, face_out, "ai"),
                                      ("portrait + C2PA record declaring AI", signed, 0.01, face_in, "ai_declared")]:
        kind, headline, _ = combine(inspect(data), p, None, face)
        print(f"{'PASS' if kind == want else 'FAIL'}  {name:46s} model {100 * p:5.1f}% -> {headline}")
        results.append(kind == want)

    assert inspect(cam)["camera"]["make"] == "TestCam"
    assert inspect(b"not an image")["notes"], "unparseable bytes should produce a note, not an exception"

    for p in map(Path, sys.argv[1:]):
        ev = inspect(p.read_bytes())
        print(f"file  {p.name:20s} camera={ev['camera']} ai_declared={ev['ai_declared']} notes={ev['notes']}")
    if not all(results):
        raise SystemExit("some checks failed")
    print("OK")


if __name__ == "__main__":
    main()
