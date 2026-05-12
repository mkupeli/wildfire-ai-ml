"""Download D-Fire dataset via Kaggle API.

Onkosul: Kaggle CLI kurulu + ~/.kaggle/kaggle.json credentials.
Manuel alternatif: https://github.com/gaiasd/DFireDataset 'tan zip indir, data/raw/dfire/ altina ac.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/dfire")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "gaiasd/dfire-dataset",
                "-p",
                str(output),
                "--unzip",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] Kaggle CLI hata: {result.stderr}", file=sys.stderr)
            print("Manuel indirme: https://github.com/gaiasd/DFireDataset", file=sys.stderr)
            sys.exit(1)
        print(f"D-Fire {output} dizinine indirildi.")
    except FileNotFoundError:
        print(
            "[ERROR] kaggle CLI bulunamadi. `pip install kaggle` veya manuel indirme.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
