"""
Digital image Degradation → Restoration → Enhancement pipeline.

Usage:
    python main.py            # all stages
    python main.py --stage 1  # a single stage
    python main.py --stage 1 2

main.py only selects and runs stages. The logic of each stage lives in
src/stages/ and writes its results to results/stageN_.../
"""
import argparse

from src.stages import STAGES


def main():
    ap = argparse.ArgumentParser(description="Image processing pipeline")
    ap.add_argument("--stage", nargs="*", type=int, default=None,
                    help="which stages to run (e.g. --stage 1 2). Without it: all.")
    args = ap.parse_args()

    todo = sorted(STAGES) if not args.stage else args.stage
    for s in todo:
        if s not in STAGES:
            print(f"[!] Stage {s} does not exist.")
            continue
        STAGES[s]()


if __name__ == "__main__":
    main()
