"""Send a Telegram message (and optionally files) to the project owner.

    python notify.py "Training finished: 98.6% accuracy"
    python notify.py "Results" --file results/spectrum.png --file results/robustness.csv

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env (never commit that file).
"""
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).parent


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"]


def send(text, files=()):
    token, chat = load_env()
    api = f"https://api.telegram.org/bot{token}"
    r = requests.post(f"{api}/sendMessage", data={"chat_id": chat, "text": text}, timeout=30)
    r.raise_for_status()
    for f in files:
        f = Path(f)
        kind = "photo" if f.suffix.lower() in {".png", ".jpg", ".jpeg"} else "document"
        with open(f, "rb") as fh:
            r = requests.post(f"{api}/send{kind.capitalize()}", data={"chat_id": chat, "caption": f.name},
                              files={kind: fh}, timeout=120)
        r.raise_for_status()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--file", action="append", default=[])
    a = ap.parse_args()
    send(a.text, a.file)
    print("sent")
