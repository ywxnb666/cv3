from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screw_seg.train import train_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics segmentation model for HW3.")
    parser.add_argument("--data_yaml", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs/train"))
    parser.add_argument("--model", default="yolo11m-seg.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_segmentation_model(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        model_name=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        batch=args.batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
