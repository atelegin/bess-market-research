"""
Pre-compute all results for this note.

Run once:  python notes/XX-slug/precompute.py
Results saved to notes/XX-slug/data/precomputed.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DATA_DIR = Path(__file__).parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: run expensive computations here
    payload = {}

    path = DATA_DIR / "precomputed.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved to {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
