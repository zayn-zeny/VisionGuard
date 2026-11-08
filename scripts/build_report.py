"""
scripts/build_report.py — compile report/report.md into
report/23L-0578_VisionGuard.pdf (via HTML + headless Microsoft Edge).

* YAML front matter becomes the title block.
* ![caption](path) becomes a numbered <figure> with the image embedded.
* Any remaining {{PLACEHOLDER}} is rendered as a visible yellow "pending"
  box, so an unfinished item can never silently ship as if it were done.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SRC = REPORT / "report.md"
HTML_OUT = REPORT / "report.html"
PDF_OUT = REPORT / "23L-0578_VisionGuard.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45;
       color: #1b1b1b; }
h1 { font-size: 17pt; border-bottom: 2px solid #334; padding-bottom: 3px; margin-top: 22px;
     page-break-before: always; }
h1.first, .titleblock + h1 { page-break-before: avoid; }
h2 { font-size: 13pt; margin-top: 18px; color: #223; }
h3 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.8pt; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
th { background: #eceef3; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f2f3f6;
       padding: 0 2px; }
pre { background: #f4f5f8; border: 1px solid #dde; padding: 6px 8px; font-size: 8pt;
      white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #99a; margin: 6px 0; padding: 2px 10px; color: #333; }
figure { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 205mm; border: 1px solid #ccc; }
figcaption { font-size: 8.8pt; color: #333; margin-top: 4px; text-align: left; }
.pending { background: #fff4c2; border: 1px solid #e0b400; padding: 6px 8px; margin: 8px 0;
           font-weight: 600; }
.titleblock { text-align: center; margin: 40mm 0 10mm; }
.titleblock .t { font-size: 24pt; font-weight: 700; }
.titleblock .s { font-size: 13pt; margin-top: 6px; }
.titleblock .a { font-size: 12pt; margin-top: 14px; }
"""

PENDING_TEXT = {
    "LIVE_API_URL": "PENDING — public API URL.",
    "CI_GREEN_SCREENSHOT": "PENDING — screenshot of the green GitHub Actions run.",
    "CI_TOY_SCREENSHOTS": "PENDING — screenshots of the red and the green toy CI runs.",
    "DEPLOY_SCREENSHOTS": ("PENDING — Vercel dashboard screenshots: project overview, "
                           "deployment build log, Settings → Environment Variables."),
    "DAGSHUB_SCREENSHOTS": ("PENDING — DagsHub repository page (Files tab) and the "
                            "DVC-tracked data/raw folder."),
    "POSTMAN_SCREENSHOTS": "PENDING — Postman screenshots against the live URL.",
    "OTHER_NETWORK": "PENDING — test from a different network (mobile hotspot).",
    "CHECK_API": "⏳ pending deployment",
    "CHECK_POSTMAN": "⏳ pending — Postman screenshots",
    "CHECK_NETWORK": "⏳ pending — hotspot test",
}


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        _, fm, text = text.split("---", 2)
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')

    fig_no = [0]

    def figure(m: re.Match) -> str:
        fig_no[0] += 1
        cap = markdown.markdown(m.group(1)).removeprefix("<p>").removesuffix("</p>")
        return (f'\n<figure><img src="{html.escape(m.group(2))}">'
                f"<figcaption><b>Figure {fig_no[0]}.</b> {cap}</figcaption></figure>\n")

    text = re.sub(r"^!\[(.+?)\]\((\S+?)\)\s*$", figure, text, flags=re.M)

    pending = []

    def placeholder(m: re.Match) -> str:
        key = m.group(1)
        pending.append(key)
        msg = PENDING_TEXT.get(key, f"PENDING — {key}")
        return msg if key.startswith("CHECK_") else f'<span class="pending">{msg}</span>'

    text = re.sub(r"\{\{([A-Z_]+)\}\}", placeholder, text)

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<h1>", '<h1 class="first">', 1)
    title = (f'<div class="titleblock"><div class="t">{meta.get("title", "")}</div>'
             f'<div class="s">{meta.get("subtitle", "")}</div>'
             f'<div class="a">{meta.get("author", "")}<br>{meta.get("date", "")}</div></div>')
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>23L-0578 VisionGuard"
           f"</title><style>{CSS}</style></head><body>{title}{body}</body></html>")
    HTML_OUT.write_text(doc, encoding="utf-8")

    profile = ROOT / "logs" / "tmp_edge_profile"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF_OUT}",
                    HTML_OUT.as_uri()], check=True, capture_output=True, timeout=300)
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(profile)], capture_output=True)

    missing = [m for m in re.findall(r'<img src="([^"]+)"', doc) if not (REPORT / m).exists()]
    print(f"figures: {fig_no[0]}  missing images: {missing or 'none'}")
    print(f"pending placeholders: {sorted(set(pending)) or 'none'}")
    print(f"wrote {PDF_OUT} ({PDF_OUT.stat().st_size / 1e6:.1f} MB)")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
