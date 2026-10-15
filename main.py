"""
Αλυσίδα Υποβάθμισης → Αποκατάστασης → Βελτίωσης ψηφιακής εικόνας.

Εκτέλεση:
    python main.py            # όλα τα στάδια
    python main.py --stage 1  # μόνο ένα στάδιο
    python main.py --stage 1 2

Το main.py μόνο επιλέγει και εκτελεί στάδια. Η λογική κάθε σταδίου βρίσκεται
στο src/stages/ και γράφει τα αποτελέσματά του στο results/stageN_.../
"""
import argparse

from src.stages import STAGES


def main():
    ap = argparse.ArgumentParser(description="Αλυσίδα επεξεργασίας εικόνας")
    ap.add_argument("--stage", nargs="*", type=int, default=None,
                    help="ποια στάδια να τρέξουν (π.χ. --stage 1 2). Χωρίς αυτό: όλα.")
    args = ap.parse_args()

    todo = sorted(STAGES) if not args.stage else args.stage
    for s in todo:
        if s not in STAGES:
            print(f"[!] Το στάδιο {s} δεν υπάρχει.")
            continue
        STAGES[s]()


if __name__ == "__main__":
    main()
