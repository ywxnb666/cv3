from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screw_seg.annotation_io import load_yolo_segmentation
from screw_seg.constants import CLASS_NAMES
from screw_seg.utils import ensure_dir, list_images, load_image, write_json, write_yaml


def annotation_difficulty(image_path: Path, label_path: Path) -> tuple[int, float]:
    image = load_image(image_path)
    height, width = image.shape[:2]
    anns = load_yolo_segmentation(label_path, width=width, height=height)
    boxes = []
    for ann in anns:
        poly = ann["polygon"]
        x1 = float(poly[:, 0].min())
        y1 = float(poly[:, 1].min())
        x2 = float(poly[:, 0].max())
        y2 = float(poly[:, 1].max())
        boxes.append((x1, y1, x2, y2))
    max_overlap = 0.0
    for i, a in enumerate(boxes):
        ax1, ay1, ax2, ay2 = a
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        if area_a <= 0:
            continue
        for b in boxes[i + 1 :]:
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            if inter <= 0:
                continue
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            if union > 0:
                max_overlap = max(max_overlap, inter / union)
    return len(anns), max_overlap


def copy_pair(image_path: Path, label_path: Path, out_images: Path, out_labels: Path) -> None:
    ensure_dir(out_images)
    ensure_dir(out_labels)
    shutil.copy2(image_path, out_images / image_path.name)
    shutil.copy2(label_path, out_labels / label_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val split with real 10/2 and optional synthetic train append.")
    parser.add_argument("--real_dataset_dir", type=Path, required=True)
    parser.add_argument("--synth_dataset_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--val_count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    real_images = list_images(args.real_dataset_dir / "images")
    scored = []
    for image_path in real_images:
        label_path = args.real_dataset_dir / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        count, overlap = annotation_difficulty(image_path, label_path)
        scored.append({"image": image_path, "label": label_path, "count": count, "overlap": overlap})
    scored.sort(key=lambda row: (row["count"], row["overlap"], row["image"].stem), reverse=True)
    val_items = scored[: args.val_count]
    train_items = scored[args.val_count :]

    out_images_train = ensure_dir(args.output_dir / "images" / "train")
    out_images_val = ensure_dir(args.output_dir / "images" / "val")
    out_labels_train = ensure_dir(args.output_dir / "labels" / "train")
    out_labels_val = ensure_dir(args.output_dir / "labels" / "val")

    for row in train_items:
        copy_pair(row["image"], row["label"], out_images_train, out_labels_train)
    for row in val_items:
        copy_pair(row["image"], row["label"], out_images_val, out_labels_val)

    synth_added = 0
    if args.synth_dataset_dir:
        for image_path in list_images(args.synth_dataset_dir / "images"):
            label_path = args.synth_dataset_dir / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            copy_pair(image_path, label_path, out_images_train, out_labels_train)
            synth_added += 1

    write_yaml(
        args.output_dir / "dataset.yaml",
        {
            "path": str(args.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
        },
    )
    write_json(
        args.output_dir / "split_report.json",
        {
            "train_real": [row["image"].name for row in train_items],
            "val_real": [row["image"].name for row in val_items],
            "val_scores": [
                {"image": row["image"].name, "instances": row["count"], "max_overlap": row["overlap"]}
                for row in val_items
            ],
            "synth_added": synth_added,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
