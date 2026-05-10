"""Training entrypoint.

Usage:
    python -m wildfire_ml.train

Implementation: Phase 1 Sprint 2.
"""

import argparse

from wildfire_ml.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train smoke detector")
    parser.add_argument("--config", type=str, default=None, help="Config file (Phase 1)")
    _args = parser.parse_args()

    cfg = Config()
    print(f"Config loaded: {cfg}")
    raise NotImplementedError("Training loop: Phase 1 Sprint 2")


if __name__ == "__main__":
    main()
