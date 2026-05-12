"""Download pyro-sdis dataset from HuggingFace cache."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/pyro-sdis")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Downloading pyro-sdis to {args.output} ...")
    load_dataset("pyronear/pyro-sdis", cache_dir=args.output)
    print("Done.")


if __name__ == "__main__":
    main()
