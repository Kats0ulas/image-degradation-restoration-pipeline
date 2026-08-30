"""
Δημιουργία πινάκων αποτελεσμάτων σε τρεις μορφές:
  - εκτύπωση στο τερματικό (ευθυγραμμισμένη)
  - αρχείο .csv  (για επεξεργασία)
  - αρχείο .md   (έτοιμο για αντιγραφή στην αναφορά)
"""
import csv
from pathlib import Path

from .io_utils import ensure_dir


def _fmt(v):
    # None = η στήλη δεν έχει νόημα για τη γραμμή· τυπώνεται ως παύλα.
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def render(headers, rows) -> str:
    """Ευθυγραμμισμένος πίνακας ως string (για το τερματικό)."""
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells)
    return f"{line}\n{sep}\n{body}"


def save(headers, rows, path_stem, title=None) -> None:
    """Αποθηκεύει τον πίνακα ως .csv και .md με βάση το path_stem (χωρίς κατάληξη)."""
    stem = Path(path_stem)
    ensure_dir(stem.parent)

    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(headers)
        w.writerows([[_fmt(c) for c in r] for r in rows])

    md = []
    if title:
        md.append(f"**{title}**\n")
    md.append("| " + " | ".join(str(h) for h in headers) + " |")
    md.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        md.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
