"""Download D-Fire dataset.

Usage:
    python scripts/download_dfire.py --output data/raw/dfire
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/dfire")
    parser.parse_args()
    raise NotImplementedError("Implement in Phase 0.5.C")


if __name__ == "__main__":
    main()
