from __future__ import annotations

import argparse
from pathlib import Path

from screw_seg.infer import InferencePipeline
from screw_seg.utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW3 screw instance segmentation inference entrypoint.")
    parser.add_argument("--data_dir", type=Path, default=Path("./data"))
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/inference"))
    parser.add_argument("--config", type=Path, default=Path("./configs/default.yaml"))
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = read_yaml((project_root / args.config).resolve() if not args.config.is_absolute() else args.config)
    pipeline = InferencePipeline(project_root=project_root, config=config)
    pipeline.run(
        data_dir=(project_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir,
        output_dir=(project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
