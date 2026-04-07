from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screw_seg.synth import generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate copy-paste synthetic dataset for screw instance segmentation.")
    parser.add_argument("--asset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=360)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_synthetic_dataset(
        asset_dir=args.asset_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
