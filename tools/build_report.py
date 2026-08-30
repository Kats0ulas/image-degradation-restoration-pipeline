"""Παράγει την τελική αναφορά: report/anafora.md -> anafora.html -> anafora.pdf

Χρήση:
    python tools/build_report.py            # HTML + PDF
    python tools/build_report.py --html     # μόνο HTML

Το PDF παράγεται με headless Chrome/Edge (print-to-pdf), οπότε δεν χρειάζεται
LaTeX ή pandoc. Οι εικόνες φορτώνονται με σχετικά paths από το results/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "report" / "anafora.md"
HTML_PATH = ROOT / "report" / "anafora.html"
PDF_PATH = ROOT / "report" / "anafora.pdf"

BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }

:root {
  --ink: #16191d;
  --muted: #5a6270;
  --rule: #c9cfd8;
  --accent: #1f4e79;
  --tint: #f2f5f9;
}

* { box-sizing: border-box; }

body {
  font-family: Georgia, "Times New Roman", "DejaVu Serif", serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: var(--ink);
  background: #ffffff;
  margin: 0 auto;
  max-width: 190mm;
  padding: 12mm 6mm;
  text-align: justify;
  hyphens: auto;
}

h1, h2, h3, h4 {
  font-family: "Segoe UI", Calibri, "DejaVu Sans", sans-serif;
  color: var(--accent);
  line-height: 1.25;
  text-align: left;
  page-break-after: avoid;
}

h1 { font-size: 23pt; margin: 0 0 6mm 0; }
h2 {
  font-size: 15pt;
  margin: 9mm 0 4mm 0;
  padding-bottom: 1.5mm;
  border-bottom: 1.2px solid var(--rule);
}
h3 { font-size: 12pt; margin: 6mm 0 2.5mm 0; }

p { margin: 0 0 3mm 0; }

strong { color: #0f1216; }

code, pre, .url {
  font-family: Consolas, "DejaVu Sans Mono", monospace;
  font-size: 9pt;
}

code {
  background: var(--tint);
  padding: 0.4mm 1mm;
  border-radius: 2px;
}

pre {
  background: var(--tint);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  padding: 3mm 4mm;
  overflow-x: auto;
  line-height: 1.4;
  page-break-inside: avoid;
}

pre code { background: none; padding: 0; }

ol, ul { margin: 0 0 3mm 0; padding-left: 7mm; }
li { margin-bottom: 1.5mm; }

/* Πίνακες */

table {
  border-collapse: collapse;
  width: 100%;
  margin: 3mm 0 5mm 0;
  font-size: 8.8pt;
  page-break-inside: avoid;
}

thead { background: var(--accent); color: #fff; }

th, td {
  border: 0.6px solid var(--rule);
  padding: 1.4mm 2mm;
  text-align: left;
  vertical-align: top;
}

th {
  font-family: "Segoe UI", Calibri, sans-serif;
  font-weight: 600;
  font-size: 8.6pt;
}

tbody tr:nth-child(even) { background: var(--tint); }

td em { color: var(--muted); font-size: 8.2pt; }

/* Εικόνες */

figure {
  margin: 5mm 0 6mm 0;
  text-align: center;
  page-break-inside: avoid;
}

figure img {
  max-width: 100%;
  max-height: 115mm;
  border: 1px solid var(--rule);
  background: #fff;
}

figcaption {
  font-size: 8.6pt;
  color: var(--muted);
  text-align: left;
  margin-top: 2mm;
  line-height: 1.45;
}

.note {
  font-size: 8.6pt;
  color: var(--muted);
  margin-top: -3mm;
}

/* Εξώφυλλο */

.cover {
  text-align: center;
  padding-top: 30mm;
  page-break-after: always;
}

.cover h1 {
  font-size: 26pt;
  text-align: center;
  margin: 10mm 0 6mm 0;
}

.cover .uni {
  font-family: "Segoe UI", Calibri, sans-serif;
  font-size: 10pt;
  letter-spacing: 0.4px;
  color: var(--muted);
  text-transform: uppercase;
}

.cover .course {
  font-family: "Segoe UI", Calibri, sans-serif;
  font-size: 13pt;
  font-weight: 600;
  color: var(--accent);
  margin-top: 3mm;
}

.cover .subtitle {
  font-size: 12pt;
  color: var(--muted);
  font-style: italic;
  margin-bottom: 25mm;
}

.cover .author {
  font-size: 11pt;
  line-height: 1.9;
  margin-top: 20mm;
}

.cover .date {
  margin-top: 18mm;
  font-size: 10pt;
  color: var(--muted);
}

/* Περιεχόμενα */

.toc ul { list-style: none; padding-left: 5mm; }
.toc > ul { padding-left: 0; }
.toc li { margin-bottom: 1.2mm; }
.toc a { color: var(--ink); text-decoration: none; }
.toc > ul > li > a { font-weight: 600; color: var(--accent); }

a { color: var(--accent); }

.pagebreak { page-break-after: always; break-after: page; }

@media print {
  body { padding: 0; max-width: none; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8">
<title>ΕΥ631 — Τελική Εργασία</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def shrink_figures(text: str, max_width: int = 1700) -> str:
    """Φτιάχνει υποδειγματοληπτημένα αντίγραφα των εικόνων στο report/figs/.

    Οι αρχικές εικόνες είναι ~2800 px πλάτος, ενώ στη σελίδα A4 τυπώνονται σε
    ~170 mm — δηλαδή ~2000 px στα 300 dpi. Η σμίκρυνση μειώνει δραστικά το
    μέγεθος του PDF χωρίς ορατή απώλεια στην εκτύπωση.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[i] Το Pillow δεν είναι εγκατεστημένο — οι εικόνες μπαίνουν σε πλήρη ανάλυση.")
        return text

    figs_dir = ROOT / "report" / "figs"
    figs_dir.mkdir(exist_ok=True)

    def repl(match: "re.Match[str]") -> str:
        rel = match.group(1)
        source = (ROOT / "report" / rel).resolve()
        if not source.exists():
            print(f"[!] Λείπει η εικόνα: {rel}")
            return match.group(0)

        target = figs_dir / f"{source.parent.name}__{source.name}"
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            with Image.open(source) as img:
                if img.width > max_width:
                    height = round(img.height * max_width / img.width)
                    img = img.resize((max_width, height), Image.LANCZOS)
                img.save(target, optimize=True)
        return f'src="figs/{target.name}"'

    return re.sub(r'src="(\.\./results/[^"]+)"', repl, text)


def build_html() -> str:
    text = MD_PATH.read_text(encoding="utf-8")
    text = shrink_figures(text)
    # Επιτρέπει markdown μέσα στο <div class="cover">
    text = text.replace('<div class="cover">', '<div class="cover" markdown="1">')

    md = markdown.Markdown(
        extensions=["tables", "toc", "attr_list", "md_in_html", "fenced_code", "sane_lists"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    body = md.convert(text)
    html = HTML_TEMPLATE.format(css=CSS, body=body)
    HTML_PATH.write_text(html, encoding="utf-8")
    return html


def find_browser() -> str | None:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("chrome") or shutil.which("msedge")


def build_pdf() -> bool:
    browser = find_browser()
    if browser is None:
        print("[!] Δεν βρέθηκε Chrome/Edge — παρήχθη μόνο το HTML.")
        print("    Άνοιξε το report/anafora.html και τύπωσέ το σε PDF (Ctrl+P).")
        return False

    if PDF_PATH.exists():
        PDF_PATH.unlink()

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--allow-file-access-from-files",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.as_uri(),
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)

    for _ in range(20):
        if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Build της τελικής αναφοράς ΕΥ631")
    parser.add_argument("--html", action="store_true", help="μόνο HTML, χωρίς PDF")
    args = parser.parse_args()

    if not MD_PATH.exists():
        print(f"[!] Δεν βρέθηκε το {MD_PATH}")
        return 1

    build_html()
    print(f"[OK] HTML : {HTML_PATH}  ({HTML_PATH.stat().st_size / 1024:.0f} KB)")

    if args.html:
        return 0

    if build_pdf():
        print(f"[OK] PDF  : {PDF_PATH}  ({PDF_PATH.stat().st_size / 1024:.0f} KB)")
        return 0

    print("[!] Η παραγωγή PDF απέτυχε — το HTML είναι έτοιμο για εκτύπωση σε PDF.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
