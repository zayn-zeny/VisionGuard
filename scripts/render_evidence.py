"""
scripts/render_evidence.py — turn real command transcripts and real
server-rendered pages into report screenshots.

* term  : renders a transcript from logs/ (each command line already carries
          "[user@host date time 23L-0578_A2_Ops]$") as a terminal-style PNG.
* page  : fetches a page from the running app (GET, or POST of a real file so
          the error/result pages are produced by the app itself), saves the
          exact HTML with a <base> tag, and screenshots it with headless Edge.
          A header strip records the URL, request, capture time, user@host and
          project name, because headless Edge has no address bar.

Usage examples:
    python scripts/render_evidence.py term logs/25_compose_healthy.log SS_38.png "compose ps"
    python scripts/render_evidence.py page GET  http://localhost:5000/history SS_35b.png
    python scripts/render_evidence.py page POST http://localhost:5000/predict SS_31b.png x.jpg
    python scripts/render_evidence.py live https://dagshub.com/<user>/<repo>.mlflow SS_19.png
"""
from __future__ import annotations

import getpass
import re
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "screenshots"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
MONO = r"C:\Windows\Fonts\consola.ttf"
SANS = r"C:\Windows\Fonts\segoeui.ttf"
STAMP_USER = f"{getpass.getuser()}@{socket.gethostname()}"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _header(img: Image.Image, text: str) -> Image.Image:
    font = ImageFont.truetype(SANS, 15)
    bar = 34
    out = Image.new("RGB", (img.width, img.height + bar), (235, 237, 240))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.text((12, 6), text, fill=(30, 30, 30), font=font)
    return out


def term(log: str, png: str, title: str, max_lines: int = 70, width: int = 150) -> None:
    lines = []
    for raw in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        raw = ANSI.sub("", raw).replace("\t", "    ")
        if not raw.strip() and lines and not lines[-1].strip():
            continue
        while len(raw) > width:
            lines.append(raw[:width])
            raw = "    " + raw[width:]
        lines.append(raw)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [f"... ({len(lines) - max_lines + 1} more lines in {log})"]
    font = ImageFont.truetype(MONO, 15)
    lh, cw = 19, 8.3
    w = int(width * cw) + 30
    h = len(lines) * lh + 24
    img = Image.new("RGB", (w, h), (18, 18, 24))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        colour = (110, 220, 140) if line.startswith("[") and "]$ " in line else (225, 225, 225)
        d.text((15, 12 + i * lh), line, fill=colour, font=font)
    img = _header(img, f"Terminal (Git Bash)  |  {title}  |  source: {log}  |  23L-0578_A2_Ops")
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / png)
    print("wrote", OUT / png)


def page(method: str, url: str, png: str, upload: str | None = None,
         ctype: str = "image/jpeg", height: int = 900) -> None:
    s = requests.Session()
    if method == "POST":
        data = Path(upload).read_bytes() if upload and Path(upload).exists() else upload.encode()
        name = Path(upload).name if upload and Path(upload).exists() else "upload.bin"
        r = s.post(url, files={"image": (name, data, ctype)}, allow_redirects=True, timeout=120)
        first = r.history[0].status_code if r.history else r.status_code
        req = f"POST {url}  file={name}  ->  HTTP {first}"
        if r.history:
            req += f" -> GET {r.url} HTTP {r.status_code}"
    else:
        r = s.get(url, timeout=60)
        req = f"GET {r.url}  ->  HTTP {r.status_code}"
    origin = re.match(r"https?://[^/]+", url).group(0)
    html = r.text
    if "<head>" in html:
        html = html.replace("<head>", f'<head><base href="{origin}/">', 1)
    else:
        html = f'<base href="{origin}/"><pre>{html}</pre>'
    tmpdir = Path(tempfile.mkdtemp(dir=str(ROOT / "logs")))
    f = tmpdir / "page.html"
    f.write_text(html, encoding="utf-8")
    shot = tmpdir / "shot.png"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={tmpdir / 'profile'}",
                    f"--window-size=1400,{height}", f"--screenshot={shot}", f.as_uri()],
                   check=True, capture_output=True, timeout=120)
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img = _header(Image.open(shot).convert("RGB"),
                  f"{req}  |  {when}  |  {STAMP_USER}  |  23L-0578_A2_Ops")
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / png)
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(tmpdir)], capture_output=True)
    print("wrote", OUT / png, "|", req)


def live(url: str, png: str, height: str = "1000", wait_ms: str = "20000") -> None:
    """Screenshot a live (JavaScript) page such as the DagsHub MLflow UI."""
    tmpdir = Path(tempfile.mkdtemp(dir=str(ROOT / "logs")))
    shot = tmpdir / "shot.png"
    budget = [f"--virtual-time-budget={wait_ms}"] if wait_ms != "0" else []
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={tmpdir / 'profile'}", *budget,
                    f"--window-size=1500,{height}", f"--screenshot={shot}", url],
                   check=False, capture_output=True, timeout=180)
    if not shot.exists():
        raise SystemExit(f"Edge produced no screenshot for {url}")
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img = _header(Image.open(shot).convert("RGB"),
                  f"GET {url}  |  {when}  |  {STAMP_USER}  |  23L-0578_A2_Ops")
    img.save(OUT / png)
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(tmpdir)], capture_output=True)
    print("wrote", OUT / png)


if __name__ == "__main__":
    kind, *args = sys.argv[1:]
    if kind == "term":
        term(*args[:3], *map(int, args[3:]))
    elif kind == "live":
        live(*args)
    elif kind == "page":
        page(*args)
    else:
        raise SystemExit(__doc__)
