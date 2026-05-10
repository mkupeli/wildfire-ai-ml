"""Download pyro-sdis dataset from Hugging Face Hub.

Usage:
    python scripts/download_pyrosdis.py --output data/raw/pyro-sdis
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/pyro-sdis")
    parser.parse_args()
    raise NotImplementedError("Implement in Phase 0.5.C")


if __name__ == "__main__":
    main()
