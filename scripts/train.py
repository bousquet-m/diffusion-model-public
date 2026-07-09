"""Train the diffusion denoiser.

    python scripts/train.py --config configs/smoke.yaml   # fast end-to-end
    python scripts/train.py --config configs/base.yaml     # real run
"""
from __future__ import annotations

import argparse

from insite_diff.config import load_config
from insite_diff.training.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
