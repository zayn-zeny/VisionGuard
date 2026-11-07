"""
scripts/test_remote_api.py — Task 7/8: exercise the deployed inference API.

Usage:
    python scripts/test_remote_api.py                 # uses REMOTE_API_URL from .env
    python scripts/test_remote_api.py https://.../predict

Reads REMOTE_API_URL and REMOTE_API_KEY from the environment (.env); the key is
never printed. Sends real AI/real test images plus four failure cases and
prints status code, latency and JSON body for each, with a timestamped header
so the transcript can be attached to the report.
"""
from __future__ import annotations

import getpass
import io
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ("AI", "data/raw/AiArtData/AiArtData/bDSF6saJUuPLsEu9i3ud--1--gvs7i.jpg"),
    ("AI", "data/raw/AiArtData/AiArtData/6324662e5ee3110438933c1b_COBE-Pink-Elephant-Midjourney-Article.jpg"),
    ("AI", "data/raw/AiArtData/AiArtData/landscape-art-ai-generated-photo2.jpg"),
    ("Real", "data/raw/RealArt/RealArt/360_F_380747975_sS1hCVB0qPqFCWBMZ3qJ5xTqH6rtaDBI.jpg"),
    ("Real", "data/raw/RealArt/RealArt/33100scr.jpg"),
    ("Real", "data/raw/RealArt/RealArt/the-essay_58_resurgence-of-the-people_2019-article-image.jpg"),
]


def _post(url: str, key: str | None, name: str, data: bytes, ctype: str):
    headers = {"X-API-Key": key} if key else {}
    t0 = time.perf_counter()
    r = requests.post(url, headers=headers,
                      files={"image": (name, io.BytesIO(data), ctype)}, timeout=120)
    ms = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except ValueError:
        body = r.text[:200]
    return r.status_code, ms, body


def main() -> None:
    load_dotenv(ROOT / ".env")
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ["REMOTE_API_URL"]
    key = os.environ.get("REMOTE_API_KEY", "")
    base = url.rsplit("/predict", 1)[0]

    print(f"VisionGuard API test | project 23L-0578_A2_Ops | user {getpass.getuser()}"
          f"@{socket.gethostname()} ({platform.system()})")
    print(f"time  : {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"target: {url}")
    try:
        ip = requests.get("https://api.ipify.org", timeout=10).text
        # masked: enough to show two runs came from different networks
        print("client public IP: " + ".".join(ip.split(".")[:2]) + ".x.x")
    except requests.RequestException:
        pass
    h = requests.get(base + "/health", timeout=120)
    print(f"GET /health -> {h.status_code} {h.text}\n")

    correct = 0
    for truth, rel in SAMPLES:
        p = ROOT / rel
        code, ms, body = _post(url, key, p.name, p.read_bytes(), "image/jpeg")
        ok = isinstance(body, dict) and body.get("label", "").startswith(truth[:2])
        correct += ok
        print(f"[{truth:4}] {p.name[:45]:45} -> {code} {ms:7.1f} ms {body}")
    print(f"\ncorrect: {correct}/{len(SAMPLES)}\n")

    img = (ROOT / SAMPLES[0][1]).read_bytes()
    cases = [
        ("missing API key", None, "a.jpg", img, "image/jpeg"),
        ("wrong file type", key, "notes.txt", b"plain text", "text/plain"),
        ("corrupt image", key, "broken.jpg", b"\xff\xd8\xff\xe0" + b"\x00garbage" * 64,
         "image/jpeg"),
        ("oversized file", key, "huge.jpg", b"\xff\xd8" + b"0" * (11 * 1024 * 1024),
         "image/jpeg"),
    ]
    for label, k, name, data, ctype in cases:
        code, ms, body = _post(url, k, name, data, ctype)
        print(f"[error] {label:16} -> {code} {ms:7.1f} ms {body}")


if __name__ == "__main__":
    main()
