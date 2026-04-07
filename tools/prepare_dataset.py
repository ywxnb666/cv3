from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screw_seg.annotation_io import convert_labelme_dir_to_yolo, extract_instance_assets
from screw_seg.utils import ensure_dir, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert instance annotations to YOLO-seg format and extract RGBA assets.")
    parser.add_argument("--labelme_dir", type=Path, required=True, help="Directory that contains Labelme json files and source images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Prepared YOLO-seg dataset output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = ensure_dir(args.output_dir)
    report = convert_labelme_dir_to_yolo(src_dir=args.labelme_dir, out_dir=dataset_dir)
    asset_report = extract_instance_assets(dataset_dir=dataset_dir, output_dir=dataset_dir / "instance_assets")
    write_json(dataset_dir / "prepare_report.json", {"dataset": report, "assets": asset_report})
    write_yaml(
        dataset_dir / "dataset.yaml",
        {
            "path": str(dataset_dir.resolve()),
            "train": "images",
            "val": "images",
            "names": {idx: name for idx, name in enumerate(["Type_1", "Type_2", "Type_3", "Type_4", "Type_5"])},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
