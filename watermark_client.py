"""Client for the external watermark check running on a second PC.

Each image opened in the app is sent there automatically; the answer comes back as
True (AI watermark found), False (none found) or None (could not be checked).

Settings in .env:
    DESK_URL=http://<second-pc-address>:8600
    DESK_TOKEN=<shared key, same value on both PCs>
"""
import os
from pathlib import Path

import requests

ROOT = Path(__file__).parent
ANSWER_TO_WATERMARK = {"found": True, "not_found": False, "could_not_check": None}


def env(name):
    if os.environ.get(name):
        return os.environ[name]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return ""


def configured():
    return bool(env("DESK_URL") and env("DESK_TOKEN"))


def _headers():
    return {"X-Desk-Token": env("DESK_TOKEN")}


def send(data, filename):
    """Send an image to the check; returns the job id."""
    r = requests.post(env("DESK_URL").rstrip("/") + "/api/jobs", headers=_headers(),
                      files={"image": (filename, data)}, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def status(job_id):
    """Returns (status, watermark): status 'pending' or 'answered'; watermark True / False / None."""
    r = requests.get(env("DESK_URL").rstrip("/") + f"/api/jobs/{job_id}", headers=_headers(), timeout=15)
    r.raise_for_status()
    job = r.json()
    return job["status"], ANSWER_TO_WATERMARK.get(job["answer"])
